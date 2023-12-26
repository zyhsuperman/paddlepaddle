import sys
import paddle
from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from models import SyncNet_color as SyncNet
import audio
import numpy as np
from glob import glob
import os, random, cv2, argparse
from hparams import hparams, get_image_list
parser = argparse.ArgumentParser(description=
    'Code to train the expert lip-sync discriminator')
parser.add_argument('--data_root', help=
    'Root folder of the preprocessed LRS2 dataset', required=True)
parser.add_argument('--checkpoint_dir', help=
    'Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help=
    'Resumed from this checkpoint', default=None, type=str)
args = parser.parse_args()
global_step = 0
global_epoch = 0
use_cuda = paddle.device.cuda.device_count() >= 1
print('use_cuda: {}'.format(use_cuda))
syncnet_T = 5
syncnet_mel_step_size = 16

paddle.device.set_device('gpu')

class Dataset(object):

    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)
        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80.0 * (start_frame_num / float(hparams.fps)))
        end_idx = start_idx + syncnet_mel_step_size
        return spec[start_idx:end_idx, :]

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)
            if random.choice([True, False]):
                y = paddle.ones(shape=[1]).astype(dtype='float32')
                chosen = img_name
            else:
                y = paddle.zeros(shape=[1]).astype(dtype='float32')
                chosen = wrong_img_name
            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                continue
            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    break
                window.append(img)
            if not all_read:
                continue
            try:
                wavpath = join(vidname, 'audio.wav')
                wav = audio.load_wav(wavpath, hparams.sample_rate)
                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue
            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            if mel.shape[0] != syncnet_mel_step_size:
                continue
            x = np.concatenate(window, axis=2) / 255.0
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1] // 2:]
            x = paddle.to_tensor(data=x, dtype='float32')
            mel = paddle.to_tensor(data=mel.T, dtype='float32').unsqueeze(axis
                =0)
            return x, mel, y


logloss = paddle.nn.BCELoss()


def cosine_loss(a, v, y):
    d = paddle.nn.functional.cosine_similarity(x1=a, x2=v)
    loss = logloss(d.unsqueeze(axis=1), y)
    return loss


def train(model, train_data_loader, test_data_loader, optimizer,
    checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step
    while global_epoch < nepochs:
        running_loss = 0.0
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y) in prog_bar:
            model.train()
            """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
# >>>>>>            optimizer.zero_grad()
            optimizer.clear_grad()
            a, v = model(mel, x)
            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()
            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()
            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, global_step,
                    checkpoint_dir, global_epoch)
            if global_step % hparams.syncnet_eval_interval == 0:
                with paddle.no_grad():
                    eval_model(test_data_loader, global_step, model,
                        checkpoint_dir)
            prog_bar.set_description('Loss: {}'.format(running_loss / (step +
                1)))
        global_epoch += 1


def eval_model(test_data_loader, global_step, model, checkpoint_dir):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):
            model.eval()
            a, v = model(mel, x)
            loss = cosine_loss(a, v, y)
            losses.append(loss.item())
            if step > eval_steps:
                break
        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)
        return


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(checkpoint_dir, 'checkpoint_step{:09d}.pdparams'.
        format(global_step))
    optimizer_state = optimizer.state_dict(
        ) if hparams.save_optimizer_state else None
    paddle.save(obj={'state_dict': model.state_dict(), 'optimizer':
        optimizer_state, 'global_step': step, 'global_epoch': epoch}, path=
        checkpoint_path)
    print('Saved checkpoint:', checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = paddle.load(path=checkpoint_path)
    else:
        checkpoint = paddle.load(path=checkpoint_path)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch
    print('Load checkpoint from: {}'.format(path))
    checkpoint = _load(path)
    model.set_state_dict(state_dict=checkpoint['state_dict'])
    if not reset_optimizer:
        optimizer_state = checkpoint['optimizer']
        if optimizer_state is not None:
            print('Load optimizer state from {}'.format(path))
            optimizer.set_state_dict(state_dict=checkpoint['optimizer'])
    global_step = checkpoint['global_step']
    global_epoch = checkpoint['global_epoch']
    return model


if __name__ == '__main__':
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    train_dataset = Dataset('test')
    test_dataset = Dataset('val')
# >>>>>>    train_data_loader = torch.utils.data.DataLoader(train_dataset,
#         batch_size=hparams.syncnet_batch_size, shuffle=True, num_workers=
#         hparams.num_workers)
# >>>>>>    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size
#         =hparams.syncnet_batch_size, num_workers=8)

    # For the training dataset
    train_data_loader = paddle.io.DataLoader(
        train_dataset, 
        batch_size=hparams.syncnet_batch_size, 
        shuffle=True, 
        num_workers=hparams.num_workers
    )

    # For the test dataset
    test_data_loader = paddle.io.DataLoader(
        test_dataset, 
        batch_size=hparams.syncnet_batch_size, 
        shuffle=False,  # Typically, the test dataset is not shuffled
        num_workers=8
    )
    device = str('cuda' if use_cuda else 'cpu').replace('cuda', 'gpu')
    model = SyncNet()
    print('total trainable params {}'.format(sum(p.size for p in model.
        parameters() if not p.stop_gradient)))
    optimizer = paddle.optimizer.Adam(parameters=[p for p in model.
        parameters() if not p.stop_gradient], learning_rate=hparams.
        syncnet_lr, weight_decay=0.0)
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=
            False)
    train(model, train_data_loader, test_data_loader, optimizer,
        checkpoint_dir=checkpoint_dir, checkpoint_interval=hparams.
        syncnet_checkpoint_interval, nepochs=hparams.nepochs)
