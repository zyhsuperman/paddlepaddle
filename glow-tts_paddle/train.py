import paddle
import os
import json
import argparse
import math
from data_utils import TextMelLoader, TextMelCollate
import models
import commons
import utils
from text.symbols import symbols
from visualdl import LogWriter
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert paddle.device.cuda.device_count(
        ) >= 1, 'CPU training is not allowed.'
    n_gpus = paddle.device.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12266'
    hps = utils.get_hparams()
    # paddle.distributed.spawn(func=train_and_eval, nprocs=n_gpus, args=(
    #     n_gpus, hps))
    train_and_eval(rank=0, n_gpus=1, hps=hps)


def train_and_eval(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = LogWriter(logdir=hps.model_dir)
        writer_eval = LogWriter(logdir=os.path.join(hps.model_dir, 'eval'))
    paddle.distributed.init_parallel_env()
    paddle.seed(seed=hps.train.seed)
    paddle.device.set_device('gpu')
    train_dataset = TextMelLoader(hps.data.training_files, hps.data)
    # train_sampler = paddle.io.DistributedBatchSampler(dataset=train_dataset,
    #     num_replicas=n_gpus, rank=rank, shuffle=True, batch_size=1)
    collate_fn = TextMelCollate(1)
    # train_loader = paddle.io.DataLoader(train_dataset, num_workers=0,
    #     shuffle=False, batch_size=hps.train.batch_size,
    #     drop_last=True, collate_fn=collate_fn, sampler=train_sampler)
    train_loader = paddle.io.DataLoader(train_dataset, num_workers=0,
        shuffle=False, batch_size=hps.train.batch_size,
        drop_last=True, collate_fn=collate_fn)
    if rank == 0:
        val_dataset = TextMelLoader(hps.data.validation_files, hps.data)
        val_loader = paddle.io.DataLoader(val_dataset, num_workers=0,
            shuffle=False, batch_size=hps.train.batch_size, 
            drop_last=True, collate_fn=collate_fn)
    generator = models.FlowGenerator(n_vocab=len(symbols) + getattr(hps.
        data, 'add_blank', False), out_channels=hps.data.n_mel_channels, **
        hps.model)
    optimizer_g = commons.Adam(generator.parameters(), scheduler=hps.train.
        scheduler, dim_model=hps.model.hidden_channels, warmup_steps=hps.
        train.warmup_steps, lr=hps.train.learning_rate, betas=hps.train.
        betas, eps=hps.train.eps)
    generator = paddle.DataParallel(layers=generator)
    epoch_str = 1
    global_step = 0
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.
            latest_checkpoint_path(hps.model_dir, 'G_*.pdparams'), generator,
            optimizer_g)
        epoch_str += 1
        optimizer_g.step_num = (epoch_str - 1) * len(train_loader)
        optimizer_g._update_learning_rate()
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        if hps.train.ddi and os.path.isfile(os.path.join(hps.model_dir,
            'ddi_G.pdparams')):
            _ = utils.load_checkpoint(os.path.join(hps.model_dir,
                'ddi_G.pdparams'), generator, optimizer_g)
    scaler = paddle.amp.GradScaler(enable=hps.train.fp16_run,
        incr_every_n_steps=2000, init_loss_scaling=65536.0)
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train(rank, epoch, hps, generator, optimizer_g, train_loader,
                logger, writer, scaler)
            evaluate(rank, epoch, hps, generator, optimizer_g, val_loader,
                logger, writer_eval)
            utils.save_checkpoint(generator, optimizer_g, hps.train.
                learning_rate, epoch, os.path.join(hps.model_dir,
                'G_{}.pdparams'.format(epoch)))
        else:
            train(rank, epoch, hps, generator, optimizer_g, train_loader,
                None, None)


def train(rank, epoch, hps, generator, optimizer_g, train_loader, logger,
    writer, scaler):
    # train_loader.set_epoch(epoch)
    global global_step
    generator.train()
    for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(train_loader):
        x, x_lengths = x, x_lengths
        y, y_lengths = y, y_lengths
        
        optimizer_g.zero_grad()
        
        if hps.train.fp16_run:
            with paddle.amp.auto_cast():
                (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn,
                    logw, logw_) = generator(x, x_lengths, y, y_lengths,
                    gen=False)
                l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
                l_length = commons.duration_loss(logw, logw_, x_lengths)
                loss_gs = [l_mle, l_length]
                loss_g = sum(loss_gs)
            scaler.scale(loss_g).backward()
            grad_norm = commons.clip_grad_value_(generator.parameters(), 5)
            scaler.step(optimizer_g)
            scaler.update()
        else:
            (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn,
                logw, logw_) = generator(x, x_lengths, y, y_lengths, gen=False)
            l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
            l_length = commons.duration_loss(logw, logw_, x_lengths)
            loss_gs = [l_mle, l_length]
            loss_g = sum(loss_gs)
            loss_g.backward()
            grad_norm = commons.clip_grad_value_(generator.parameters(), 5)
            optimizer_g.step()
        if rank == 0:
            if batch_idx % hps.train.log_interval == 0:
                (y_gen, *_), *_ = generator(x[:1], x_lengths[:1],
                    gen=True)
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                    .format(epoch, batch_idx * len(x), len(train_loader.
                    dataset), 100.0 * batch_idx / len(train_loader), loss_g
                    .item()))
                logger.info([x.item() for x in loss_gs] + [global_step,
                    optimizer_g.get_lr()])
                scalar_dict = {'loss/g/total': loss_g, 'learning_rate':
                    optimizer_g.get_lr(), 'grad_norm': grad_norm}
                scalar_dict.update({'loss/g/{}'.format(i): v for i, v in
                    enumerate(loss_gs)})
                utils.summarize(writer=writer, global_step=global_step,
                    images={'y_org': utils.plot_spectrogram_to_numpy(y[0].
                    data.cpu().numpy()), 'y_gen': utils.
                    plot_spectrogram_to_numpy(y_gen[0].data.cpu().numpy()),
                    'attn': utils.plot_alignment_to_numpy(attn[0, 0].data.
                    cpu().numpy())}, scalars=scalar_dict)
        global_step += 1
    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger,
    writer_eval):
    if rank == 0:
        global global_step
        generator.eval()
        losses_tot = []
        with paddle.no_grad():
            for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(val_loader
                ):
                x, x_lengths = x, x_lengths
                y, y_lengths = y, y_lengths
                (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn,
                    logw, logw_) = generator(x, x_lengths, y, y_lengths,
                    gen=False)
                l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
                l_length = commons.duration_loss(logw, logw_, x_lengths)
                loss_gs = [l_mle, l_length]
                loss_g = sum(loss_gs)
                if batch_idx == 0:
                    losses_tot = loss_gs
                else:
                    losses_tot = [(x + y) for x, y in zip(losses_tot, loss_gs)]
                if batch_idx % hps.train.log_interval == 0:
                    logger.info(
                        'Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                        format(epoch, batch_idx * len(x), len(val_loader.
                        dataset), 100.0 * batch_idx / len(val_loader),
                        loss_g.item()))
                    logger.info([x.item() for x in loss_gs])
        losses_tot = [(x / len(val_loader)) for x in losses_tot]
        loss_tot = sum(losses_tot)
        scalar_dict = {'loss/g/total': loss_tot}
        scalar_dict.update({'loss/g/{}'.format(i): v for i, v in enumerate(
            losses_tot)})
        utils.summarize(writer=writer_eval, global_step=global_step,
            scalars=scalar_dict)
        logger.info('====> Epoch: {}'.format(epoch))


if __name__ == '__main__':
    main()
