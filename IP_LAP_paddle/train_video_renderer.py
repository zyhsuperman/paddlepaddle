import paddle
from os.path import join, isfile
from tqdm import tqdm
from tensorboardX import SummaryWriter
from paddle.vision.transforms import functional as F
import numpy as np
from glob import glob
import os, random
import cv2
from piq import psnr, ssim, FID
import face_alignment
from piq.feature_extractors import InceptionV3
from models import define_D
from loss import GANLoss
from models import Renderer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sketch_root', required=True, help=
    'root path for sketches')
parser.add_argument('--face_img_root', required=True, help=
    'root path for face frame images')
parser.add_argument('--audio_root', required=True, help=
    'root path for audio mel')
args = parser.parse_args()
paddle.device.set_device('gpu:0')
num_workers = 0
Project_name = 'renderer_T1_ref_N3'
finetune_path = None
ref_N = 3
T = 1
print('Project_name:', Project_name)
batch_size = 16
batch_size_val = 16
mel_step_size = 16
fps = 25
img_size = 128
FID_batch_size = 1024
evaluate_interval = 1500
checkpoint_interval = evaluate_interval
lr = 0.0001
global_step, global_epoch = 0, 0
sketch_root = args.sketch_root
face_img_root = args.face_img_root
filelist_name = 'lrs2'
audio_root = args.audio_root
checkpoint_root = './checkpoints/renderer/'
checkpoint_dir = os.path.join(checkpoint_root, 'Pro_' + Project_name)
reset_optimizer = False
save_optimizer_state = True
writer = SummaryWriter('tensorboard_runs/Project_{}'.format(Project_name))
criterionFeat = paddle.nn.L1Loss()


class Dataset(object):

    def get_vid_name_list(self, split):
        filelist = []
        with open('filelists/{}/{}.txt'.format(filelist_name, split)) as f:
            for line in f:
                line = line.strip()
                if ' ' in line:
                    line = line.split()[0]
                filelist.append(line)
        return filelist

    def __init__(self, split):
        min_len = 25
        vid_name_lists = self.get_vid_name_list(split)
        self.available_video_names = []
        print('filter videos with min len of ', min_len, '....')
        for vid_name in tqdm(vid_name_lists, total=len(vid_name_lists)):
            img_paths = list(glob(join(face_img_root, vid_name, '*.png')))
            vid_len = len(img_paths)
            if vid_len >= min_len:
                self.available_video_names.append((vid_name, vid_len))
        print('complete,with available vids: ', len(self.
            available_video_names), '\n')

    def normalize_and_transpose(self, window):
        x = np.asarray(window) / 255.0
        x = np.transpose(x, (0, 3, 1, 2))
        return paddle.to_tensor(data=x, dtype='float32')

    def __len__(self):
        return len(self.available_video_names)

    def __getitem__(self, idx):
        while 1:
            vid_idx = random.randint(0, len(self.available_video_names) - 1)
            vid_name = self.available_video_names[vid_idx][0]
            vid_len = self.available_video_names[vid_idx][1]
            face_img_paths = list(glob(join(face_img_root, vid_name, '*.png')))
            window_T = 5
            random_start_idx = random.randint(0, vid_len - window_T)
            T_idxs = list(range(random_start_idx, random_start_idx + window_T))
            T_face_paths = [os.path.join(face_img_root, vid_name, str(idx) +
                '.png') for idx in T_idxs]
            ref_N_fpaths = random.sample(face_img_paths, ref_N)
            T_frame_img = []
            T_frame_sketch = []
            for img_path in T_face_paths:
                sketch_path = os.path.join(sketch_root, '/'.join(img_path.
                    split('/')[-3:]))
                if os.path.isfile(img_path) and os.path.isfile(sketch_path):
                    T_frame_img.append(cv2.resize(cv2.imread(img_path), (
                        img_size, img_size)))
                    T_frame_sketch.append(cv2.imread(sketch_path))
                else:
                    break
            if len(T_frame_img) != window_T:
                continue
            ref_N_frame_img, ref_N_frame_sketch = [], []
            for img_path in ref_N_fpaths:
                sketch_path = os.path.join(sketch_root, '/'.join(img_path.
                    split('/')[-3:]))
                if os.path.isfile(img_path) and os.path.isfile(sketch_path):
                    ref_N_frame_img.append(cv2.resize(cv2.imread(img_path),
                        (img_size, img_size)))
                    ref_N_frame_sketch.append(cv2.imread(sketch_path))
                else:
                    break
            if len(ref_N_frame_img) != ref_N:
                continue
            T_frame_img = self.normalize_and_transpose(T_frame_img)
            T_frame_sketch = self.normalize_and_transpose(T_frame_sketch)
            ref_N_frame_img = self.normalize_and_transpose(ref_N_frame_img)
            ref_N_frame_sketch = self.normalize_and_transpose(
                ref_N_frame_sketch)
            try:
                audio_mel = np.load(join(audio_root, vid_name, 'audio.npy'))
            except Exception as e:
                continue
            frame_idx = T_idxs[2]
            mel_start_frame_idx = frame_idx - 2
            if mel_start_frame_idx < 0:
                continue
            start_idx = int(80.0 * (mel_start_frame_idx / float(fps)))
            m = audio_mel[start_idx:start_idx + mel_step_size, :]
            if m.shape[0] != mel_step_size:
                continue
            T_mels = m.T
            T_mels = paddle.to_tensor(data=T_mels, dtype='float32').unsqueeze(
                axis=0).unsqueeze(axis=0)
            return T_frame_img[2].unsqueeze(axis=0
                ), T_frame_sketch, ref_N_frame_img, ref_N_frame_sketch, T_mels


def load_checkpoint(path, model, optimizer, reset_optimizer=False,
    overwrite_global_states=True):
    global global_step
    global global_epoch
    print('Load checkpoint from: {}'.format(path))
    checkpoint = paddle.load(path=path)
    s = checkpoint['state_dict']
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '', 1)] = v
    model.set_state_dict(state_dict=new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint['optimizer']
        if optimizer_state is not None:
            print('Load optimizer state from {}'.format(path))
            optimizer.set_state_dict(state_dict=checkpoint['optimizer'])
    if overwrite_global_states:
        global_step = checkpoint['global_step']
        global_epoch = checkpoint['global_epoch']
    return model


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(checkpoint_dir,
        '{}_epoch_{}_checkpoint_step{:09d}.pdparams'.format(prefix, epoch,
        global_step))
    if isfile(checkpoint_path):
        os.remove(checkpoint_path)
    optimizer_state = optimizer.state_dict() if save_optimizer_state else None
    paddle.save(obj={'state_dict': model.state_dict(), 'optimizer':
        optimizer_state, 'global_step': step, 'global_epoch': epoch}, path=
        checkpoint_path)
    print('Saved checkpoint:', checkpoint_path)


n_layers_D = 3
num_D = 2
disc = define_D(input_nc=3, ndf=64, n_layers_D=n_layers_D, norm='instance',
    use_sigmoid=False, num_D=num_D, getIntermFeat=True)
criterionGAN = GANLoss(use_lsgan=True, tensor=paddle.Tensor)
fid_metric = FID()
feature_extractor = InceptionV3()

def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis]//num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)

def compute_generation_quality(gt, fake_image):
    global global_step
    import torch
    fake_image_numpy = fake_image.cpu().numpy()
    gt_numpy = gt.cpu().numpy()

    fake_image_torch = torch.tensor(fake_image_numpy).to('cuda')
    gt_torch = torch.tensor(gt_numpy).to('cuda')
    psnr_values = []
    ssim_values = []
    psnr_value = psnr(fake_image_torch, gt_torch, reduction='none')
    psnr_values.extend([e.item() for e in psnr_value])
    ssim_value = ssim(fake_image_torch, gt_torch, data_range=1.0, reduction='none')
    ssim_values.extend([e.item() for e in ssim_value])
    return np.asarray(psnr_values).mean(), np.asarray(ssim_values).mean()


def save_sample_images_gen(T_frame_sketch, ref_N_frame_img, wrapped_ref,
    generated_img, gt, global_step, checkpoint_dir):
    ref_N_frame_img = ref_N_frame_img.unsqueeze(axis=1).expand(shape=[-1, T,
        -1, -1, -1, -1])
    ref_N_frame_img = (ref_N_frame_img.cpu().numpy().transpose(0, 1, 2, 4, 
        5, 3) * 255.0).astype(np.uint8)
    fake_image = paddle.stack(x=split(x=generated_img,
        num_or_sections=T, axis=0), axis=0)
    fake_image = (fake_image.detach().cpu().numpy().transpose(0, 1, 3, 4, 2
        ) * 255.0).astype(np.uint8)
    wrapped_ref = paddle.stack(x=split(x=wrapped_ref,
        num_or_sections=T, axis=0), axis=0)
    wrapped_ref = (wrapped_ref.detach().cpu().numpy().transpose(0, 1, 3, 4,
        2) * 255.0).astype(np.uint8)
    gt = paddle.stack(x=split(x=gt, num_or_sections=T, axis=0),
        axis=0)
    gt = (gt.cpu().numpy().transpose(0, 1, 3, 4, 2) * 255.0).astype(np.uint8)
    T_frame_sketch = (T_frame_sketch[:, (2)].unsqueeze(axis=1).cpu().numpy(
        ).transpose(0, 1, 3, 4, 2) * 255.0).astype(np.uint8)
    folder = join(checkpoint_dir, 'samples_step{:09d}'.format(global_step))
    if not os.path.exists(folder):
        os.mkdir(folder)
    collage = np.concatenate((T_frame_sketch, *[ref_N_frame_img[:, :, (i)] for
        i in range(ref_N_frame_img.shape[2])], wrapped_ref, fake_image, gt),
        axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.png'.format(folder, batch_idx, t), c[t])


def evaluate(model, val_data_loader):
    global global_epoch, global_step
    eval_epochs = 1
    print('Evaluating model for {} epochs'.format(eval_epochs))
    eval_warp_loss, eval_gen_loss = 0.0, 0.0
    count = 0
    psnrs, ssims, fids = [], [], []
    for epoch in range(eval_epochs):
        prog_bar = tqdm(enumerate(val_data_loader), total=len(val_data_loader))
        for step, (T_frame_img, T_frame_sketch, ref_N_frame_img,
            ref_N_frame_sketch, T_mels) in prog_bar:
            model.eval()
            (T_frame_img, T_frame_sketch, ref_N_frame_img,
                ref_N_frame_sketch, T_mels) = (T_frame_img, T_frame_sketch,
                ref_N_frame_img, ref_N_frame_sketch, T_mels)
            (generated_img, wrapped_ref, perceptual_warp_loss,
                perceptual_gen_loss) = (model(T_frame_img, T_frame_sketch,
                ref_N_frame_img, ref_N_frame_sketch, T_mels))
            perceptual_warp_loss = perceptual_warp_loss.sum()
            perceptual_gen_loss = perceptual_gen_loss.sum()
            gt = paddle.concat(x=[T_frame_img[i] for i in range(T_frame_img
                .shape[0])], axis=0)
            eval_warp_loss += perceptual_warp_loss.item()
            eval_gen_loss += perceptual_gen_loss.item()
            count += 1
            # psnr, ssim, fid = compute_generation_quality(gt, generated_img)
            # psnrs.append(psnr)
            # ssims.append(ssim)
            # fids.append(fid)

            psnr, ssim = compute_generation_quality(gt, generated_img)
            psnrs.append(psnr)
            ssims.append(ssim)
        save_sample_images_gen(T_frame_sketch, ref_N_frame_img, wrapped_ref,
            generated_img, gt, global_step, checkpoint_dir)
    # psnr, ssim, fid = np.asarray(psnrs).mean(), np.asarray(ssims).mean(
    #     ), np.asarray(fids).mean()
    # print('psnr %.3f ssim %.3f fid %.3f' % (psnr, ssim, fid))
    # writer.add_scalar('psnr', psnr, global_step)
    # writer.add_scalar('ssim', ssim, global_step)
    # writer.add_scalar('fid', fid, global_step)

    psnr, ssim = np.asarray(psnrs).mean(), np.asarray(ssims).mean(
        )
    print('psnr %.3f ssim %.3f' % (psnr, ssim))
    writer.add_scalar('psnr', psnr, global_step)
    writer.add_scalar('ssim', ssim, global_step)
    writer.add_scalar('eval_warp_loss', eval_warp_loss / count, global_step)
    writer.add_scalar('eval_gen_loss', eval_gen_loss / count, global_step)
    print('eval_warp_loss :', eval_warp_loss / count, 'eval_gen_loss', 
        eval_gen_loss / count, 'global_step:', global_step)


if __name__ == '__main__':
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    device = str('cuda').replace('cuda', 'gpu')
    model = Renderer()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
        learning_rate=lr, weight_decay=0.0)
    if finetune_path is not None:
        load_checkpoint(finetune_path, model, optimizer, reset_optimizer=
            False, overwrite_global_states=False)

    disc = disc
    disc_optimizer = paddle.optimizer.Adam(parameters=[p for p in disc.
        parameters() if not p.stop_gradient], learning_rate=0.0001, beta1=(
        0.5, 0.999)[0], beta2=(0.5, 0.999)[1], weight_decay=0.0)
    train_dataset = Dataset('train')
    val_dataset = Dataset('test')

    train_data_loader = paddle.io.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=num_workers,
                                            use_shared_memory=True)

    val_data_loader = paddle.io.DataLoader(val_dataset,
                                        batch_size=batch_size_val,
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=num_workers,
                                        use_shared_memory=True)

    while global_epoch < 9999999999:
        prog_bar = tqdm(enumerate(train_data_loader), total=len(
            train_data_loader))
        running_warp_loss, running_gen_loss = 0.0, 0.0
        for step, (T_frame_img, T_frame_sketch, ref_N_frame_img,
            ref_N_frame_sketch, T_mels) in prog_bar:
            model.train()
            disc.train()
            optimizer.clear_grad()
            disc_optimizer.clear_grad()
            (T_frame_img, T_frame_sketch, ref_N_frame_img,
                ref_N_frame_sketch, T_mels) = (T_frame_img, T_frame_sketch,
                ref_N_frame_img, ref_N_frame_sketch, T_mels)
            (generated_img, wrapped_ref, perceptual_warp_loss,
                perceptual_gen_loss) = (model(T_frame_img, T_frame_sketch,
                ref_N_frame_img, ref_N_frame_sketch, T_mels))
            perceptual_warp_loss = perceptual_warp_loss.sum()
            perceptual_gen_loss = perceptual_gen_loss.sum()
            gt = paddle.concat(x=[T_frame_img[i] for i in range(T_frame_img
                .shape[0])], axis=0)
            pred_fake = disc.forward(generated_img.detach())
            loss_D_fake = criterionGAN(pred_fake, False)
            pred_real = disc.forward(gt.clone().detach())
            loss_D_real = criterionGAN(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real).mean() * 0.5
            pred_fake = disc.forward(generated_img)
            loss_G_GAN = criterionGAN(pred_fake, True).mean()
            loss_G_GAN_Feat = 0
            feat_weights = 4.0 / (n_layers_D + 1)
            D_weights = 1.0 / num_D
            for i in range(num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += (D_weights * feat_weights *
                        criterionFeat(pred_fake[i][j], pred_real[i][j].
                        detach()).mean() * 2.5)
            if global_epoch > 25:
                loss = (2.5 * perceptual_warp_loss + 4 *
                    perceptual_gen_loss + 0.1 * 2.5 * loss_G_GAN +
                    loss_G_GAN_Feat)
            else:
                loss = 2.5 * perceptual_warp_loss + 0 * perceptual_gen_loss
            loss.backward()
            optimizer.step()
            loss_D.backward()
            disc_optimizer.step()
            running_warp_loss += perceptual_warp_loss.item()
            running_gen_loss += perceptual_gen_loss.item()
            if global_step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, global_step,
                    checkpoint_dir, global_epoch, prefix=Project_name)
            if (global_step % evaluate_interval == 0 or global_step == 100 or
                global_step == 500):
                with paddle.no_grad():
                    evaluate(model, val_data_loader)
            prog_bar.set_description(
                'epoch: %d step: %d running_warp_loss: %.4f running_gen_loss: %.4f'
                 % (global_epoch, global_step, running_warp_loss / (step + 
                1), running_gen_loss / (step + 1)))
            writer.add_scalar('running_warp_loss', running_warp_loss / (
                step + 1), global_step)
            writer.add_scalar('running_gen_loss', running_gen_loss / (step +
                1), global_step)
            global_step += 1
        global_epoch += 1
print('end')
