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


class FlowGenerator_DDI(models.FlowGenerator):
    """A helper for Data-dependent Initialization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for f in self.decoder.flows:
            if getattr(f, 'set_ddi', False):
                f.set_ddi(True)


def main():
    paddle.set_device('gpu')
    hps = utils.get_hparams()
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    paddle.seed(seed=hps.train.seed)
    train_dataset = TextMelLoader(hps.data.training_files, hps.data)
    collate_fn = TextMelCollate(1)
    train_loader = paddle.io.DataLoader(train_dataset, num_workers=0,
        shuffle=True, batch_size=hps.train.batch_size, 
        drop_last=True, collate_fn=collate_fn)
    generator = FlowGenerator_DDI(len(symbols) + getattr(hps.data,
        'add_blank', False), out_channels=hps.data.n_mel_channels, **hps.model)
    optimizer_g = commons.Adam(generator.parameters(), scheduler=hps.train.
        scheduler, dim_model=hps.model.hidden_channels, warmup_steps=hps.
        train.warmup_steps, lr=hps.train.learning_rate, betas=hps.train.
        betas, eps=hps.train.eps)
    generator.train()
    for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(train_loader):
        x, x_lengths = x, x_lengths
        y, y_lengths = y, y_lengths
        _ = generator(x, x_lengths, y, y_lengths, gen=False)
        break
    utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, 
        0, os.path.join(hps.model_dir, 'ddi_G.pdparams'))


if __name__ == '__main__':
    main()
