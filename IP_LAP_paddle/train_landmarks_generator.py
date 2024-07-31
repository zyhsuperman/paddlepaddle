import paddle
import os
from tqdm import tqdm
import numpy as np
from glob import glob
from os.path import join, isfile
import random
from tensorboardX import SummaryWriter
from models import Landmark_generator as Landmark_transformer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pre_audio_root', default=
    '...../Dataset/lrs2_preprocessed_audio', help=
    'root path for preprocessed  audio')
parser.add_argument('--landmarks_root', default=
    '...../Dataset/lrs2_landmarks', help=
    'root path for preprocessed  landmarks')
args = parser.parse_args()
paddle.device.set_device('gpu:0')
d_model = 512
dim_feedforward = 1024
nlayers = 4
nhead = 4
dropout = 0.1
Nl = 15
T = 5
Project_name = 'landmarkT5_d512_fe1024_lay4_head4'
print('Project_name:', Project_name)
finetune_path = None
num_workers = 8
batch_size = 128
batch_size_val = 128
evaluate_interval = 5000
checkpoint_interval = evaluate_interval
mel_step_size = 16
fps = 25
lr = 0.0001
global_step, global_epoch = 0, 0
landmark_root = args.landmarks_root
filelist_name = 'lrs2'
checkpoint_root = './checkpoints/landmark_generation/'
checkpoint_dir = os.path.join(checkpoint_root, 'Pro_' + Project_name)
reset_optimizer = False
save_optimizer_state = True
paddle.device.set_device('gpu:0')
writer = SummaryWriter('tensorboard_runs/Project_{}'.format(Project_name))
ori_sequence_idx = [162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 
    148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 336, 296, 334, 293, 300, 276,
    283, 282, 295, 285, 168, 6, 197, 195, 5, 48, 115, 220, 45, 4, 275, 440,
    344, 278, 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 
    145, 144, 163, 7, 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390,
    373, 374, 380, 381, 382, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 78, 191, 80, 81, 82, 13,
    312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
full_face_sequence = [*list(range(0, 4)), *list(range(21, 25)), *list(range
    (25, 91)), *list(range(4, 21)), *list(range(91, 131))]


class LandmarkDict(dict):

    def __init__(self, idx, x, y):
        self['idx'] = idx
        self['x'] = x
        self['y'] = y

    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class Dataset(object):

    def get_vidname_list(self, split):
        vid_name_list = []
        with open('filelists/{}/{}.txt'.format(filelist_name, split)) as f:
            for line in f:
                line = line.strip()
                if ' ' in line:
                    line = line.split()[0]
                vid_name_list.append(line)
        return vid_name_list

    def __init__(self, split):
        min_len = 25
        vid_name_lists = self.get_vidname_list(split)
        self.all_video_names = []
        print('init dataset,filtering very short videos.....')
        for vid_name in tqdm(vid_name_lists, total=len(vid_name_lists)):
            pkl_paths = list(glob(join(landmark_root, vid_name, '*.npy')))
            vid_len = len(pkl_paths)
            if vid_len >= min_len:
                self.all_video_names.append((vid_name, vid_len))
        print('complete,with available vids: ', len(self.all_video_names), '\n'
            )

    def __len__(self):
        return len(self.all_video_names)

    def __getitem__(self, idx):
        while 1:
            vid_idx = random.randint(0, len(self.all_video_names) - 1)
            vid_name = self.all_video_names[vid_idx][0]
            vid_len = self.all_video_names[vid_idx][1]
            random_start_idx = random.randint(2, vid_len - T - 2)
            T_idxs = list(range(random_start_idx, random_start_idx + T))
            all_list = [i for i in range(vid_len) if i not in T_idxs]
            Nl_idxs = random.sample(all_list, Nl)
            Nl_landmarks_paths = [os.path.join(landmark_root, vid_name, str
                (idx) + '.npy') for idx in Nl_idxs]
            Nl_pose_landmarks, Nl_content_landmarks = [], []
            for frame_landmark_path in Nl_landmarks_paths:
                if not os.path.exists(frame_landmark_path):
                    break
                landmarks = np.load(frame_landmark_path, allow_pickle=True
                    ).item()
                Nl_pose_landmarks.append(landmarks['pose_landmarks'])
                Nl_content_landmarks.append(landmarks['content_landmarks'])
            if len(Nl_pose_landmarks) != Nl:
                continue
            Nl_pose = paddle.zeros(shape=(Nl, 2, 74))
            Nl_content = paddle.zeros(shape=(Nl, 2, 57))
            for idx in range(Nl):
                Nl_pose_landmarks[idx] = sorted(Nl_pose_landmarks[idx], key
                    =lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
                Nl_content_landmarks[idx] = sorted(Nl_content_landmarks[idx
                    ], key=lambda land_tuple: ori_sequence_idx.index(
                    land_tuple[0]))
                Nl_pose[(idx), (0), :] = paddle.to_tensor(data=[
                    Nl_pose_landmarks[idx][i][1] for i in range(len(
                    Nl_pose_landmarks[idx]))], dtype='float32')
                Nl_pose[(idx), (1), :] = paddle.to_tensor(data=[
                    Nl_pose_landmarks[idx][i][2] for i in range(len(
                    Nl_pose_landmarks[idx]))], dtype='float32')
                Nl_content[(idx), (0), :] = paddle.to_tensor(data=[
                    Nl_content_landmarks[idx][i][1] for i in range(len(
                    Nl_content_landmarks[idx]))], dtype='float32')
                Nl_content[(idx), (1), :] = paddle.to_tensor(data=[
                    Nl_content_landmarks[idx][i][2] for i in range(len(
                    Nl_content_landmarks[idx]))], dtype='float32')
            T_ladnmark_paths = [os.path.join(landmark_root, vid_name, str(
                idx) + '.npy') for idx in T_idxs]
            T_pose_landmarks, T_content_landmarks = [], []
            for frame_landmark_path in T_ladnmark_paths:
                if not os.path.exists(frame_landmark_path):
                    break
                landmarks = np.load(frame_landmark_path, allow_pickle=True
                    ).item()
                T_pose_landmarks.append(landmarks['pose_landmarks'])
                T_content_landmarks.append(landmarks['content_landmarks'])
            if len(T_pose_landmarks) != T:
                continue
            T_pose = paddle.zeros(shape=(T, 2, 74))
            T_content = paddle.zeros(shape=(T, 2, 57))
            for idx in range(T):
                T_pose_landmarks[idx] = sorted(T_pose_landmarks[idx], key=
                    lambda land_tuple: ori_sequence_idx.index(land_tuple[0]))
                T_content_landmarks[idx] = sorted(T_content_landmarks[idx],
                    key=lambda land_tuple: ori_sequence_idx.index(
                    land_tuple[0]))
                T_pose[(idx), (0), :] = paddle.to_tensor(data=[
                    T_pose_landmarks[idx][i][1] for i in range(len(
                    T_pose_landmarks[idx]))], dtype='float32')
                T_pose[(idx), (1), :] = paddle.to_tensor(data=[
                    T_pose_landmarks[idx][i][2] for i in range(len(
                    T_pose_landmarks[idx]))], dtype='float32')
                T_content[(idx), (0), :] = paddle.to_tensor(data=[
                    T_content_landmarks[idx][i][1] for i in range(len(
                    T_content_landmarks[idx]))], dtype='float32')
                T_content[(idx), (1), :] = paddle.to_tensor(data=[
                    T_content_landmarks[idx][i][2] for i in range(len(
                    T_content_landmarks[idx]))], dtype='float32')
            try:
                audio_mel = np.load(join(args.pre_audio_root, vid_name,
                    'audio.npy'))
            except Exception as e:
                continue
            T_mels = []
            for frame_idx in T_idxs:
                mel_start_frame_idx = frame_idx - 2
                if mel_start_frame_idx < 0:
                    break
                start_idx = int(80.0 * (mel_start_frame_idx / float(fps)))
                m = audio_mel[start_idx:start_idx + mel_step_size, :]
                if m.shape[0] != mel_step_size:
                    break
                T_mels.append(m.T)
            if len(T_mels) != T:
                continue
            T_mels = np.asarray(T_mels)
            T_mels = paddle.to_tensor(data=T_mels, dtype='float32').unsqueeze(
                axis=1)
            return T_mels, T_pose, T_content, Nl_pose, Nl_content


def load_checkpoint(path, model, optimizer, reset_optimizer=False,
    overwrite_global_states=True):
    global global_step
    global global_epoch
    print('Load checkpoint from: {}'.format(path))
    checkpoint = paddle.load(path=path)
    s = checkpoint['state_dict']
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
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


criterion_L1 = paddle.nn.L1Loss()

def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis]//num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)


def get_velocity_loss(pred, gt):
    pred = paddle.stack(x=split(x=pred, num_or_sections=T, axis=
        0), axis=0)
    gt = paddle.stack(x=split(x=gt, num_or_sections=T, axis=0),
        axis=0)
    pred = paddle.concat(x=[pred[:, :, :, (i)] for i in range(pred.shape[3]
        )], axis=2)
    gt = paddle.concat(x=[gt[:, :, :, (i)] for i in range(gt.shape[3])], axis=2
        )
    b, t, c = pred.shape
    pred_spiky = pred[:, 1:, :] - pred[:, :-1, :]
    gt_spiky = gt[:, 1:, :] - gt[:, :-1, :]
    pred_spiky = pred_spiky.reshape([b * (t - 1), c])
    gt_spiky = gt_spiky.reshape([b * (t - 1), c])
    pairwise_distance = paddle.nn.functional.pairwise_distance(x=pred_spiky,
        y=gt_spiky)
    return paddle.mean(x=pairwise_distance)


def evaluate(model, val_data_loader):
    global global_epoch, global_step
    eval_epochs = 25
    print('Evaluating model for {} epochs'.format(eval_epochs))
    eval_L1_loss = 0.0
    eval_velocity_loss = 0.0
    count = 0
    folder = join(checkpoint_dir, 'samples_step{:09d}'.format(global_step))
    if not os.path.exists(folder):
        os.mkdir(folder)
    for epoch in tqdm(range(eval_epochs), total=eval_epochs):
        prog_bar = enumerate(val_data_loader)
        for step, (T_mels, T_pose, T_content, Nl_pose, Nl_content) in prog_bar:
            model.eval()
            T_mels, T_pose, T_content, Nl_pose, Nl_content = (T_mels,
                T_pose, T_content, Nl_pose, Nl_content)
            predict_content = model(T_mels, T_pose, Nl_pose, Nl_content)
            T_content = paddle.concat(x=[T_content[i] for i in range(
                T_content.shape[0])], axis=0)
            eval_L1_loss += criterion_L1(predict_content, T_content).item()
            eval_velocity_loss += get_velocity_loss(predict_content, T_content
                ).item()
            count += 1
    writer.add_scalar('eval_L1_loss', eval_L1_loss / count, global_step)
    print('eval_L1_loss', eval_L1_loss / count, 'global_step:', global_step)
    writer.add_scalar('eval_velocity_loss', eval_velocity_loss / count,
        global_step)
    print('eval_velocity_loss', eval_velocity_loss / count, 'global_step:',
        global_step)


if __name__ == '__main__':
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    device = str('cuda').replace('cuda', 'gpu')
    model = Landmark_transformer(T, d_model, nlayers, nhead,
        dim_feedforward, dropout)
    if finetune_path is not None:
        model_dict = model.state_dict()
        print('load module....from :', finetune_path)
        checkpoint = paddle.load(path=finetune_path)
        s = checkpoint['state_dict']
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        state_dict_needed = {k: v for k, v in new_s.items() if k in
            model_dict.keys()}
        model_dict.update(state_dict_needed)
        model.set_state_dict(state_dict=model_dict)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
        learning_rate=lr, weight_decay=0.0)
    model = model
    train_dataset = Dataset('train')
    val_dataset = Dataset('test')


    train_data_loader = paddle.io.DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=num_workers,
                                use_shared_memory=True)  # `pin_memory` 在 Paddle 中对应为 `use_shared_memory`

    val_data_loader = paddle.io.DataLoader(val_dataset,
                                batch_size=batch_size_val,
                                shuffle=True,
                                drop_last=True,
                                num_workers=num_workers,
                                use_shared_memory=True)
    while global_epoch < 9999999999:
        prog_bar = tqdm(enumerate(train_data_loader), total=len(
            train_data_loader))
        running_L1_loss, running_velocity_loss = 0.0, 0.0
        for step, (T_mels, T_pose, T_content, Nl_pose, Nl_content) in prog_bar:
            if global_step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, global_step,
                    checkpoint_dir, global_epoch, prefix=Project_name)
            if global_step % evaluate_interval == 0 or global_step == 100:
                with paddle.no_grad():
                    evaluate(model, val_data_loader)
            T_mels, T_pose, T_content, Nl_pose, Nl_content = (T_mels,
                T_pose, T_content, Nl_pose, Nl_content)
            model.train()
            optimizer.clear_grad()
            predict_content = model(T_mels, T_pose, Nl_pose, Nl_content)
            T_content = paddle.concat(x=[T_content[i] for i in range(
                T_content.shape[0])], axis=0)
            L1_loss = criterion_L1(predict_content, T_content)
            Velocity_loss = get_velocity_loss(predict_content, T_content)
            loss = L1_loss + Velocity_loss
            loss.backward()
            optimizer.step()
            running_L1_loss += L1_loss.item()
            running_velocity_loss += Velocity_loss.item()
            prog_bar.set_description(
                'epoch: %d step: %d running_L1_loss: %.4f  running_velocity_loss: %.4f '
                % (global_epoch, global_step, running_L1_loss / (step + 1),
                running_velocity_loss / (step + 1)))
            writer.add_scalar('running_L1_loss', running_L1_loss / (step + 
                1), global_step)
            writer.add_scalar('running_velocity_loss', 
                running_velocity_loss / (step + 1), global_step)
            global_step += 1
        global_epoch += 1
    print('end')
