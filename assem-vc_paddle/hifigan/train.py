import sys
sys.path.append('/home1/zhaoyh/paddlemodel/assem-vc_paddle/utils')
import paddle_aux
import paddle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import tqdm
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
False = True


def train(rank, a, h):
    if h.num_gpus > 1:
        paddle.distributed.init_parallel_env()
    paddle.seed(seed=h.seed)
    device = str('cuda:{:d}'.format(rank)).replace('cuda', 'gpu')
    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print('checkpoints directory : ', a.checkpoint_path)
    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.set_state_dict(state_dict=state_dict_g['generator'])
        mpd.set_state_dict(state_dict=state_dict_do['mpd'])
        msd.set_state_dict(state_dict=state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
    if h.num_gpus > 1:
        generator = paddle.DataParallel(layers=generator).to(device)
        mpd = paddle.DataParallel(layers=mpd).to(device)
        msd = paddle.DataParallel(layers=msd).to(device)
    optim_g = paddle.optimizer.AdamW(parameters=generator.parameters(),
        learning_rate=h.learning_rate, beta1=[h.adam_b1, h.adam_b2][0],
        beta2=[h.adam_b1, h.adam_b2][1], weight_decay=0.0)
    optim_d = paddle.optimizer.AdamW(parameters=itertools.chain(msd.
        parameters(), mpd.parameters()), learning_rate=h.learning_rate,
        beta1=[h.adam_b1, h.adam_b2][0], beta2=[h.adam_b1, h.adam_b2][1],
        weight_decay=0.0)
    if state_dict_do is not None:
        optim_g.set_state_dict(state_dict=state_dict_do['optim_g'])
        optim_d.set_state_dict(state_dict=state_dict_do['optim_d'])
    tmp_lr = paddle.optimizer.lr.ExponentialDecay(gamma=h.lr_decay,
        last_epoch=last_epoch, learning_rate=optim_g.get_lr())
    optim_g.set_lr_scheduler(tmp_lr)
    scheduler_g = tmp_lr
    tmp_lr = paddle.optimizer.lr.ExponentialDecay(gamma=h.lr_decay,
        last_epoch=last_epoch, learning_rate=optim_d.get_lr())
    optim_d.set_lr_scheduler(tmp_lr)
    scheduler_d = tmp_lr
    training_filelist, validation_filelist = get_dataset_filelist(a)
    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.
        num_mels, h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax,
        n_cache_reuse=0, shuffle=False if h.num_gpus > 1 else True,
        fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
        base_mels_path=a.input_mels_dir)
    train_sampler = paddle.io.DistributedBatchSampler(dataset=trainset,
        shuffle=True, batch_size=1) if h.num_gpus > 1 else None
>>>>>>    train_loader = torch.utils.data.DataLoader(trainset, num_workers=h.
        num_workers, shuffle=False, sampler=train_sampler, batch_size=h.
        batch_size, pin_memory=True, drop_last=True)
    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft,
            h.num_mels, h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.
            fmax, False, False, n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
            device=device, fine_tuning=a.fine_tuning, base_mels_path=a.
            input_mels_dir)
>>>>>>        validation_loader = torch.utils.data.DataLoader(validset,
            num_workers=1, shuffle=False, sampler=None, batch_size=1,
            pin_memory=True, drop_last=True)
>>>>>>        sw = torch.utils.tensorboard.SummaryWriter(os.path.join(a.
            checkpoint_path, 'logs'))
    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print('Epoch: {}'.format(epoch + 1))
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch
>>>>>>            x = torch.autograd.Variable(x.to(device, non_blocking=True))
>>>>>>            y = torch.autograd.Variable(y.to(device, non_blocking=True))
>>>>>>            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True)
                )
            y = y.unsqueeze(axis=1)
            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(axis=1), h.n_fft,
                h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin,
                h.fmax_for_loss)
            """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
>>>>>>            optim_d.zero_grad()
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                y_df_hat_r, y_df_hat_g)
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g)
            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            optim_d.step()
            """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
>>>>>>            optim_g.zero_grad()
            loss_mel = paddle.nn.functional.l1_loss(input=y_mel, label=
                y_g_hat_mel) * 45
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = (loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f +
                loss_mel)
            loss_gen_all.backward()
            optim_g.step()
            if rank == 0:
                if steps % a.stdout_interval == 0:
                    with paddle.no_grad():
                        mel_error = paddle.nn.functional.l1_loss(input=
                            y_mel, label=y_g_hat_mel).item()
                    print(
                        'Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'
                        .format(steps, loss_gen_all, mel_error, time.time() -
                        start_b))
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = '{}/g_{:08d}'.format(a.
                        checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, {'generator': (
                        generator.module if h.num_gpus > 1 else generator).
                        state_dict()})
                    checkpoint_path = '{}/do_{:08d}'.format(a.
                        checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, {'mpd': (mpd.module if
                        h.num_gpus > 1 else mpd).state_dict(), 'msd': (msd.
                        module if h.num_gpus > 1 else msd).state_dict(),
                        'optim_g': optim_g.state_dict(), 'optim_d': optim_d
                        .state_dict(), 'steps': steps, 'epoch': epoch})
                if steps % a.summary_interval == 0:
                    sw.add_scalar('training/gen_loss_total', loss_gen_all,
                        steps)
                    sw.add_scalar('training/mel_spec_error', mel_error, steps)
                if steps % a.validation_interval == 0:
                    generator.eval()
                    paddle.device.cuda.empty_cache()
                    val_err_tot = 0
                    with paddle.no_grad():
                        for j, batch in enumerate(tqdm.tqdm(
                            validation_loader, desc='Validating...')):
                            x, y, _, y_mel = batch
                            y_g_hat = generator(x.to(device))
>>>>>>                            y_mel = torch.autograd.Variable(y_mel.to(device,
                                non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(
                                axis=1), h.n_fft, h.num_mels, h.
                                sampling_rate, h.hop_size, h.win_size, h.
                                fmin, h.fmax_for_loss)
                            val_err_tot += paddle.nn.functional.l1_loss(input
                                =y_mel, label=y_g_hat_mel).item()
                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0],
                                        steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j),
                                        plot_spectrogram(x[0]), steps)
                                sw.add_audio('generated/y_hat_{}'.format(j),
                                    y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.
                                    squeeze(axis=1), h.n_fft, h.num_mels, h
                                    .sampling_rate, h.hop_size, h.win_size,
                                    h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.
                                    format(j), plot_spectrogram(y_hat_spec.
                                    squeeze(axis=0).cpu().numpy()), steps)
                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar('validation/mel_spec_error', val_err,
                            steps)
                    generator.train()
            steps += 1
        scheduler_g.step()
        scheduler_d.step()
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                int(time.time() - start)))


def main():
    print('Initializing Training Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default=
        'LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default=
        'LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    a = parser.parse_args()
    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)
    paddle.seed(seed=h.seed)
    if paddle.device.cuda.device_count() >= 1:
        paddle.seed(seed=h.seed)
        h.num_gpus = paddle.device.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass
    if h.num_gpus > 1:
        paddle.distributed.spawn(func=train, nprocs=h.num_gpus, args=(a, h))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
