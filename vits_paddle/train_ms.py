import paddle
import os
import json
import argparse
import itertools
import math
import commons
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate, DistributedBucketSampler
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols
from visualdl import LogWriter
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert paddle.device.cuda.device_count(
        ) >= 1, 'CPU training is not allowed.'
    n_gpus = paddle.device.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '80000'
    hps = utils.get_hparams()
    paddle.distributed.spawn(func=run, nprocs=n_gpus, args=(n_gpus, hps))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = LogWriter(logdir=hps.model_dir)
        writer_eval = LogWriter(logdir=os.path
            .join(hps.model_dir, 'eval'))
    paddle.distributed.init_parallel_env()
    paddle.seed(seed=hps.train.seed)
    paddle.device.set_device(device=rank)
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(train_dataset, hps.train.
        batch_size, [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus, rank=rank, shuffle=True)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = paddle.io.DataLoader(train_dataset, num_workers=8,
        shuffle=False, pin_memory=True, collate_fn=collate_fn,
        batch_sampler=train_sampler)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files,
            hps.data)
        eval_loader = paddle.io.DataLoader(eval_dataset, num_workers
            =8, shuffle=False, batch_size=hps.train.batch_size, pin_memory=
            True, drop_last=False, collate_fn=collate_fn)
    net_g = SynthesizerTrn(len(symbols), hps.data.filter_length // 2 + 1, 
        hps.train.segment_size // hps.data.hop_length, n_speakers=hps.data.
        n_speakers, **hps.model).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = paddle.optimizer.AdamW(parameters=net_g.parameters(),
        learning_rate=hps.train.learning_rate, epsilon=hps.train.eps, beta1
        =hps.train.betas[0], beta2=hps.train.betas[1], weight_decay=0.0)
    optim_d = paddle.optimizer.AdamW(parameters=net_d.parameters(),
        learning_rate=hps.train.learning_rate, epsilon=hps.train.eps, beta1
        =hps.train.betas[0], beta2=hps.train.betas[1], weight_decay=0.0)
    net_g = paddle.DataParallel(layers=net_g)
    net_d = paddle.DataParallel(layers=net_d)
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.
            latest_checkpoint_path(hps.model_dir, 'G_*.pth'), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(utils.
            latest_checkpoint_path(hps.model_dir, 'D_*.pth'), net_d, optim_d)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0
    tmp_lr = paddle.optimizer.lr.ExponentialDecay(gamma=hps.train.lr_decay,
        last_epoch=epoch_str - 2, learning_rate=optim_g.get_lr())
    optim_g.set_lr_scheduler(tmp_lr)
    scheduler_g = tmp_lr
    tmp_lr = paddle.optimizer.lr.ExponentialDecay(gamma=hps.train.lr_decay,
        last_epoch=epoch_str - 2, learning_rate=optim_d.get_lr())
    optim_d.set_lr_scheduler(tmp_lr)
    scheduler_d = tmp_lr
    scaler = paddle.amp.GradScaler(enable=hps.train.fp16_run,
        incr_every_n_steps=2000, init_loss_scaling=65536.0)
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g,
                optim_d], [scheduler_g, scheduler_d], scaler, [train_loader,
                eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g,
                optim_d], [scheduler_g, scheduler_d], scaler, [train_loader,
                None], None, None)
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler,
    loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers
    train_loader.batch_sampler.set_epoch(epoch)
    global global_step
    net_g.train()
    net_d.train()
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers
        ) in enumerate(train_loader):
        x, x_lengths = x, x_lengths
        spec, spec_lengths = spec, spec_lengths
        y, y_lengths = y, y_lengths
        speakers = speakers
        with paddle.amp.auto_cast(enable=hps.train.fp16_run):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p,
                logs_p, m_q, logs_q) = net_g(x, x_lengths, spec,
                spec_lengths, speakers)
            mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.
                n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin,
                hps.data.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.
                segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(axis=1), hps.
                data.filter_length, hps.data.n_mel_channels, hps.data.
                sampling_rate, hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax)
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length,
                hps.train.segment_size)
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with paddle.amp.auto_cast(enable=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        optim_d.clear_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)
        with paddle.amp.auto_cast(enable=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with paddle.amp.auto_cast(enable=False):
                loss_dur = paddle.sum(x=l_length.astype(dtype='float32'))
                loss_mel = paddle.nn.functional.l1_loss(input=y_mel, label=
                    y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask
                    ) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = (loss_gen + loss_fm + loss_mel + loss_dur +
                    loss_kl)
        optim_g.clear_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur,
                    loss_kl]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(epoch, 100.0 *
                    batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])
                scalar_dict = {'loss/g/total': loss_gen_all, 'loss/d/total':
                    loss_disc_all, 'learning_rate': lr, 'grad_norm_d':
                    grad_norm_d, 'grad_norm_g': grad_norm_g}
                scalar_dict.update({'loss/g/fm': loss_fm, 'loss/g/mel':
                    loss_mel, 'loss/g/dur': loss_dur, 'loss/g/kl': loss_kl})
                scalar_dict.update({'loss/g/{}'.format(i): v for i, v in
                    enumerate(losses_gen)})
                scalar_dict.update({'loss/d_r/{}'.format(i): v for i, v in
                    enumerate(losses_disc_r)})
                scalar_dict.update({'loss/d_g/{}'.format(i): v for i, v in
                    enumerate(losses_disc_g)})
                image_dict = {'slice/mel_org': utils.
                    plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    'slice/mel_gen': utils.plot_spectrogram_to_numpy(
                    y_hat_mel[0].data.cpu().numpy()), 'all/mel': utils.
                    plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    'all/attn': utils.plot_alignment_to_numpy(attn[0, 0].
                    data.cpu().numpy())}
                utils.summarize(writer=writer, global_step=global_step,
                    images=image_dict, scalars=scalar_dict)
            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(net_g, optim_g, hps.train.
                    learning_rate, epoch, os.path.join(hps.model_dir,
                    'G_{}.pth'.format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.
                    learning_rate, epoch, os.path.join(hps.model_dir,
                    'D_{}.pth'.format(global_step)))
        global_step += 1
    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with paddle.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths,
            speakers) in enumerate(eval_loader):
            x, x_lengths = x, x_lengths
            spec, spec_lengths = spec, spec_lengths
            y, y_lengths = y, y_lengths
            speakers = speakers
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            speakers = speakers[:1]
            break
        y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths,
            speakers, max_len=1000)
        y_hat_lengths = mask.sum(axis=[1, 2]).astype(dtype='int64'
            ) * hps.data.hop_length
        mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.
            n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.
            data.mel_fmax)
        y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(axis=1).astype(
            dtype='float32'), hps.data.filter_length, hps.data.
            n_mel_channels, hps.data.sampling_rate, hps.data.hop_length,
            hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
    image_dict = {'gen/mel': utils.plot_spectrogram_to_numpy(y_hat_mel[0].
        cpu().numpy())}
    audio_dict = {'gen/audio': y_hat[(0), :, :y_hat_lengths[0]]}
    if global_step == 0:
        image_dict.update({'gt/mel': utils.plot_spectrogram_to_numpy(mel[0]
            .cpu().numpy())})
        audio_dict.update({'gt/audio': y[(0), :, :y_lengths[0]]})
    utils.summarize(writer=writer_eval, global_step=global_step, images=
        image_dict, audios=audio_dict, audio_sampling_rate=hps.data.
        sampling_rate)
    generator.train()


if __name__ == '__main__':
    main()
