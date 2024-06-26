from __future__ import print_function
import argparse
import random
import paddle
from paddle.vision import transforms
from paddle.io import DataLoader
import paddle.nn.functional as F
from dataloader import *
from nets.net_trans import MMIL_Net
from utils.eval_metrics import segment_level, event_level
import pandas as pd
from ipdb import set_trace
from gpuinfo import GPUInfo
import seaborn as sns
import os

os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
paddle.seed(seed=0)
np.random.seed(0)
random.seed(0)
paddle.seed(seed=0)
paddle.seed(seed=0)



class LabelSmoothingLoss(paddle.nn.Layer):

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.cls = classes
        self.dim = dim
        self.tmp = smoothing

    def forward(self, pred, gt):
        pa, pv = gt

        pv = paddle.cast(pv != 0, 'float64') 
        pa = paddle.cast(pa != 0, 'float64')
        pred = F.log_softmax(pred, axis=self.dim)
        my_gt = paddle.matmul(pa, paddle.transpose(pv, [1, 0]))
        my_gt = paddle.clip(my_gt, max=1) 
        my_gt = my_gt / (paddle.sum(my_gt, axis=-1, keepdim=True) + 1e-10)


        return paddle.mean(x=paddle.sum(x=-my_gt * pred, axis=self.dim))


def sample_mixing_neg(pos_idx, target):
    n_batch = target.shape[0]
    rand_idx_neg = []
    for i in pos_idx:
        while True:
            j = random.randint(0, n_batch - 1)
            if i != j:
                rand_idx_neg.append(j)
                break
    return paddle.to_tensor(data=rand_idx_neg, dtype='int64')


def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    target_total = 0
    weight_a_tt = []
    weight_v_tt = []
    criterion2 = LabelSmoothingLoss(10, smoothing=args.tmp)
    for batch_idx, sample in enumerate(train_loader):
        audio, video, video_st, target = sample['audio'], sample[
            'video_s'], sample['video_st'], sample[
            'label'].astype(paddle.float32)
        v_refine = sample['v_refine']
        a_refine = sample['a_refine']
        """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        optimizer.clear_grad()
        sample_idx = paddle.arange(start=0, end=target.shape[0], dtype='int64')
        rand_idx = sample_mixing_neg(sample_idx, target)
        if args.augment:
            audio = paddle.concat(x=(audio, audio[sample_idx]), axis=0)
            video = paddle.concat(x=(video, video[rand_idx]), axis=0)
            video_st = paddle.concat(x=(video_st, video_st[rand_idx]), axis=0)
            a_refine_aug = paddle.concat(x=(a_refine, a_refine[sample_idx]),
                axis=0)
            v_refine_aug = paddle.concat(x=(v_refine, v_refine[rand_idx]),
                axis=0)
            output, a_prob, v_prob, all_prob, sims = model(audio, video,
                video_st, args, a_refine=a_refine_aug, v_refine=
                v_refine_aug, rand_idx=rand_idx, sample_idx=sample_idx,
                target=target)
        output.clip_(min=1e-07, max=1 - 1e-07)
        a_prob.clip_(min=1e-07, max=1 - 1e-07)
        v_prob.clip_(min=1e-07, max=1 - 1e-07)
        if epoch > args.init_epoch:
            a = args.audio_smoothing
            v = args.vis_smoothing
        else:
            a = args.before_audio_smoothing
            v = args.before_vis_smoothing
        if target.shape[0] == args.batch_size or True:
            Pa = a * a_refine + (1 - a) * 0.5
            Pv = v * v_refine + (1 - v) * 0.5
            label_all = paddle.logical_or(x=a_refine, y=v_refine).astype(dtype
                ='float32')
            if args.augment:
                loss_a = criterion(a_prob[:Pa.shape[0]], Pa.astype(dtype=
                    'float32'))
                loss_v = criterion(v_prob[:Pa.shape[0]], Pv.astype(dtype=
                    'float32'))
                loss_mil = criterion(output[:Pa.shape[0]], label_all)
                Pa_aug = a * a_refine_aug + (1 - a) * 0.5
                Pv_aug = v * v_refine_aug + (1 - v) * 0.5
                loss_cm = criterion2(sims, (Pa_aug, Pv_aug))
            else:
                loss_a = criterion(a_prob, Pa.astype(dtype='float32'))
                loss_v = criterion(v_prob, Pv.astype(dtype='float32'))
                loss_mil = criterion(output, label_all)
                loss_cm = criterion2(sims, (Pa, Pv))
            if epoch > args.init_epoch:
                loss = loss_mil + loss_a + loss_v + args.delta * loss_cm
            else:
                loss = loss_mil + loss_a + loss_v
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_total: {:.6f} Loss_a: {:.6f} Loss_v: {:.6f} Loss_mil: {:.6f} Loss_cm: {:.6f}'
                .format(epoch, batch_idx * len(audio), len(train_loader.
                dataset), 100.0 * batch_idx / len(train_loader), loss.item(
                ), loss_a.item(), loss_v.item(), loss_mil.item(), loss_cm.
                item()))


def eval(model, val_loader, set, args):
    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat',
        'Frying_(food)', 'Basketball_bounce', 'Fire_alarm', 'Chainsaw',
        'Cello', 'Banjo', 'Singing', 'Chicken_rooster', 'Violin_fiddle',
        'Vacuum_cleaner', 'Baby_laughter', 'Accordion', 'Lawn_mower',
        'Motorcycle', 'Helicopter', 'Acoustic_guitar',
        'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender', 'Clapping']
    model.eval()
    df = pd.read_csv(set, header=0, sep='\t')
    df_a = pd.read_csv('data/AVVP_eval_audio.csv', header=0, sep='\t')
    df_v = pd.read_csv('data/AVVP_eval_visual.csv', header=0, sep='\t')
    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []
    FN_a = 0
    FN_v = 0
    FN_av = 0
    FP_a = 0
    FP_v = 0
    FP_av = 0
    TP_a_only = 0
    TP_v_only = 0
    GT_a_only = 0
    GT_v_only = 0
    GTA_total = 0
    GTV_total = 0
    GTAV_total = 0
    class_TP_a = 0
    class_FN_a = 0
    class_FP_a = 0
    class_TP_v = 0
    class_FN_v = 0
    class_FP_v = 0
    GT_total = paddle.zeros(shape=[25])
    total_cases = 0
    total_audio = []
    total_visual = []
    total_audio_label = []
    total_visual_label = []
    total_audio_name = []
    gg_name = []
    gg_audio = []
    gg_vis = []
    yy_audio = []
    yy_vis = []
    with paddle.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio, video, video_st, target = sample['audio'], sample['video_s'], sample['video_st'], sample['label']
            output, a_prob, v_prob, frame_prob, (x1, x2) = model(audio,
                video, video_st, args)
            total_audio.append(x1)
            total_visual.append(x2)
            o = (output.cpu().detach().numpy() >= 0.5).astype(np.int_)
            Pa = frame_prob[0, :, 0, :].cpu().detach().numpy()
            Pv = frame_prob[0, :, 1, :].cpu().detach().numpy()
            Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)
            Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)
            GT_a = np.zeros((25, 10))
            GT_v = np.zeros((25, 10))
            # df_vid_a = df_a.loc[df_a['filename'] == df.loc[batch_idx, :][0]]
            filename = df.iloc[batch_idx, 0] 
            df_vid_a = df_a.loc[df_a['filename'] == filename]
            filenames = df_vid_a['filename']
            events = df_vid_a['event_labels']
            onsets = df_vid_a['onset']
            offsets = df_vid_a['offset']
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_a.index[i]])
                    x2 = int(offsets[df_vid_a.index[i]])
                    event = events[df_vid_a.index[i]]
                    idx = id_to_idx[event]
                    GT_a[idx, x1:x2] = 1
            # df_vid_v = df_v.loc[df_v['filename'] == df.loc[batch_idx, :][0]]
            filename = df.iloc[batch_idx, 0]
            df_vid_v = df_v.loc[df_v['filename'] == filename]
            filenames = df_vid_v['filename']
            events = df_vid_v['event_labels']
            onsets = df_vid_v['onset']
            offsets = df_vid_v['offset']
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_v.index[i]])
                    x2 = int(offsets[df_vid_v.index[i]])
                    event = events[df_vid_v.index[i]]
                    idx = id_to_idx[event]
                    GT_v[idx, x1:x2] = 1
            GT_av = GT_a * GT_v
            GTA_total = GTA_total + GT_a.sum(axis=-1)
            GTV_total = GTV_total + GT_v.sum(axis=-1)
            GTAV_total = GTAV_total + GT_av.sum(axis=-1)
            for time_idx in range(10):
                total_audio_label.append(str(np.array(categories)[GT_a[:,
                    time_idx] != 0]).replace('[', '').replace(']', '').
                    replace("'", ''))
                total_visual_label.append(str(np.array(categories)[GT_v[:,
                    time_idx] != 0]).replace('[', '').replace(']', '').
                    replace("'", ''))
            SO_a = np.transpose(Pa)
            SO_v = np.transpose(Pv)
            SO_av = SO_a * SO_v
            gg_name.append(sample['name'])
            gg_audio.append(paddle.to_tensor(data=SO_a))
            gg_vis.append(paddle.to_tensor(data=SO_v))
            yy_audio.append(paddle.to_tensor(data=GT_a))
            yy_vis.append(paddle.to_tensor(data=GT_v))
            f_a, f_v, f, f_av, (fn_a, fp_a), (fn_v, fp_v), (fn_av, fp_av), (
                tp_a_only, gt_a_only, tp_v_only, gt_v_only), (raw_TP_a,
                raw_FN_a, raw_FP_a, raw_TP_v, raw_FN_v, raw_FP_v
                ) = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            class_TP_a += raw_TP_a
            class_FN_a += raw_FN_a
            class_FP_a += raw_FP_a
            class_TP_v += raw_TP_v
            class_FN_v += raw_FN_v
            class_FP_v += raw_FP_v
            FN_a += fn_a
            FN_v += fn_v
            FN_av += fn_av
            FP_a += fp_a
            FP_v += fp_v
            FP_av += fp_av
            TP_a_only += tp_a_only
            TP_v_only += tp_v_only
            GT_a_only += gt_a_only
            GT_v_only += gt_v_only
            F_seg_a.append(f_a)
            F_seg_v.append(f_v)
            F_seg.append(f)
            F_seg_av.append(f_av)
            f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v,
                GT_av)
            F_event_a.append(f_a)
            F_event_v.append(f_v)
            F_event.append(f)
            F_event_av.append(f_av)
    total_audio = paddle.stack(total_audio, axis=0).squeeze(1)
    total_audio = paddle.reshape(total_audio, [total_audio.shape[0] * 10, -1])

    total_visual = paddle.stack(total_visual, axis=0).squeeze(1)
    total_visual = paddle.reshape(total_visual, [total_visual.shape[0] * 10, -1])
    FP_a_rate = FP_a / (FP_a + FN_a)
    FP_v_rate = FP_v / (FP_v + FN_v)
    FP_av_rate = FP_av / (FP_av + FN_av)
    FN_a_rate = FN_a / (FP_a + FN_a)
    FN_v_rate = FN_v / (FP_v + FN_v)
    FN_av_rate = FN_av / (FP_av + FN_av)
    TP_a_only_rate = TP_a_only / GT_a_only
    TP_v_only_rate = TP_v_only / GT_v_only
    per_class_audio = 2 * class_TP_a / (2 * class_TP_a + (class_FN_a +
        class_FP_a))
    per_class_visual = 2 * class_TP_v / (2 * class_TP_v + (class_FN_v +
        class_FP_v))
    print('Audio Event Detection Segment-level F1: {:.1f}'.format(100 * np.
        mean(np.array(F_seg_a))))
    print('Visual Event Detection Segment-level F1: {:.1f}'.format(100 * np
        .mean(np.array(F_seg_v))))
    print('Audio-Visual Event Detection Segment-level F1: {:.1f}'.format(
        100 * np.mean(np.array(F_seg_av))))
    avg_type = (100 * np.mean(np.array(F_seg_av)) + 100 * np.mean(np.array(F_seg_a)) + 100 * np.mean(np.array(F_seg_v))) / 3.0
    avg_event = 100 * np.mean(np.array(F_seg))
    print('Segment-level Type@Avg. F1: {:.1f}'.format(avg_type))
    print('Segment-level Event@Avg. F1: {:.1f}'.format(avg_event))
    print('Audio Event Detection Event-level F1: {:.1f}'.format(100 * np.
        mean(np.array(F_event_a))))
    print('Visual Event Detection Event-level F1: {:.1f}'.format(100 * np.
        mean(np.array(F_event_v))))
    print('Audio-Visual Event Detection Event-level F1: {:.1f}'.format(100 *
        np.mean(np.array(F_event_av))))
    avg_type_event = (100 * np.mean(np.array(F_event_av)) + 100 * np.
        mean(np.array(F_event_a)) + 100 * np.mean(np.array(
        F_event_v))) / 3.0
    avg_event_level = 100 * np.mean(np.array(F_event))
    print('Event-level Type@Avg. F1: {:.1f}'.format(avg_type_event))
    print('Event-level Event@Avg. F1: {:.1f}'.format(avg_event_level))
    return avg_type, avg_type_event, 100 * np.mean(np.array(F_seg_a)
        ), 100 * np.mean(np.array(F_seg_v)), 100 * np.mean(np.
        array(F_seg_av))


def main():
    from base_options import BaseOptions
    args = BaseOptions().parse()
    mygpu = GPUInfo.get_info()[0]
    gpu_source = {}
    for info in mygpu:
        if mygpu[info][0] in gpu_source.keys():
            gpu_source[mygpu[info][0]] += 1
        else:
            gpu_source[mygpu[info][0]] = 1
    for gpu_id in args.gpu:
        if gpu_id != ',':
            if gpu_id not in gpu_source.keys():
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
                break
            elif gpu_source[gpu_id] < 2:
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
                break
    if args.model == 'MMIL_Net':
        model = MMIL_Net(args)
    else:
        raise 'not recognized'
    if args.mode == 'train':
        train_dataset = LLP_dataset(label=args.label_train, audio_dir=args.
            audio_dir, video_dir=args.video_dir, st_dir=args.st_dir,
            transform=transforms.Compose([ToTensor()]))
        val_dataset = LLP_dataset(label=args.label_test, audio_dir=args.
            audio_dir, video_dir=args.video_dir, st_dir=args.st_dir,
            transform=transforms.Compose([ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=16, use_shared_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
            num_workers=0, use_shared_memory=True)
        optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
            learning_rate=args.lr, weight_decay=0.0)
        tmp_lr = paddle.optimizer.lr.StepDecay(step_size=int(args.decay_epoch),
            gamma=args.decay, learning_rate=optimizer.get_lr())
        optimizer.set_lr_scheduler(tmp_lr)
        scheduler = tmp_lr
        criterion = paddle.nn.BCELoss()
        best_F = 0
        count = 0
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            scheduler.step()
            F_seg, F_event, f_a, f_v, f_av = eval(model, val_loader, args.
                label_test, args)
            count += 1
            if F_event >= best_F:
                count = 0
                best_F = F_event
                print('#################### save model #####################')
                # paddle.save(model.state_dict(), args.model_save_dir + args.checkpoint + "_%0.2f.pdparams"%(best_F))
                paddle.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pdparams")
            if count == args.early_stop:
                exit()
    elif args.mode == 'val':
        test_dataset = LLP_dataset(label=args.label_val, audio_dir=args.
            audio_dir, video_dir=args.video_dir, st_dir=args.st_dir,
            transform=transforms.Compose([ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True)
        model.set_state_dict(state_dict=paddle.load(path=args.
            model_save_dir + args.checkpoint + '.pdparams'))
        eval(model, test_loader, args.label_val, args)
    else:
        test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.
            audio_dir, video_dir=args.video_dir, st_dir=args.st_dir,
            transform=transforms.Compose([ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True)
        model.set_state_dict(state_dict=paddle.load(path=args.
            model_save_dir + args.checkpoint + '.pdparams'))
        eval(model, test_loader, args.label_test, args)


if __name__ == '__main__':
    main()
