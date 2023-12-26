import paddle
import os
from argparse import ArgumentParser
import random
import numpy as np
from omegaconf import OmegaConf
from synthesizer import Synthesizer
from modules import VCDecoder, ResidualEncoder
from datasets import TextMelDataset, text_mel_collate
import json
from tqdm import tqdm


class MySynthesizerTrainer:

    def __init__(self, hparams, model):
        super().__init__()
        self.hparams = hparams
        hp_global = OmegaConf.load(hparams.config[0])
        hp_vc = OmegaConf.load(hparams.config[1])
        hp = OmegaConf.merge(hp_global, hp_vc)
        self.hp = hp
        self.num_speakers = len(self.hp.data.speakers)
        self.model = model
        self.global_step = 0

    def training_step(self, batch):
        (text, mel_source, speakers, input_lengths, output_lengths,
            max_input_len) = batch
        with paddle.no_grad():
            ling_s, _ = self.model(text, mel_source, input_lengths,
                output_lengths, max_input_len)
        z_s = self.model.speaker(speakers)
        mask = self.model.get_cnn_mask(output_lengths)
        residual = self.model.residual_encoder(mel_source, mask, output_lengths
            )
        ling_s = paddle.concat(x=(ling_s, residual), axis=1)
        mel_s_s = self.model.decoder(ling_s, z_s, mask)
        loss_rec = paddle.nn.functional.mse_loss(input=mel_s_s, label=
            mel_source)
        return {'loss': loss_rec}

    def validation_step(self, batch):
        (text, mel_source, speakers, input_lengths, output_lengths,
            max_input_len) = batch
        ling_s, alignment = self.model(text, mel_source, input_lengths,
            output_lengths, max_input_len)
        z_s = self.model.speaker(speakers)
        mask = self.model.get_cnn_mask(output_lengths)
        residual = self.model.residual_encoder(mel_source, mask, output_lengths
            )
        ling_s = paddle.concat(x=(ling_s, residual), axis=1)
        mel_s_s = self.model.decoder(ling_s, z_s, mask)
        loss_rec = paddle.nn.functional.mse_loss(input=mel_s_s, label=
            mel_source)
        target_speakers = np.random.choice(self.hp.data.speakers, size=
            ling_s.shape[0])
        z_t_index = paddle.to_tensor(data=[self.hp.data.speakers.index(x) for
            x in target_speakers], dtype='int64')
        z_t = self.model.speaker(z_t_index)
        mel_s_t = self.model.decoder(ling_s, z_t, mask)
        return {'loss_rec': loss_rec}

    def validation_end(self, outputs):
        loss_rec = paddle.stack(x=[x['loss_rec'] for x in outputs]).mean()
        return {'val_loss': loss_rec}

    def configure_optimizers(self):
        optimizer = paddle.optimizer.Adam(parameters=list(self.model.
            decoder.parameters()) + list(self.model.residual_encoder.
            parameters()) + list(self.model.speaker.parameters()),
            learning_rate=self.hp.train.adam.lr, weight_decay=self.hp.train
            .adam.weight_decay)
        return optimizer

    def train_dataloader(self):
        trainset = TextMelDataset(self.hp, self.hp.data.train_dir, self.hp.
            data.train_meta, train=True, norm=True)
        return paddle.io.DataLoader(trainset, batch_size=self.hp.
            train.batch_size, shuffle=True, num_workers=self.hp.train.
            num_workers, collate_fn=text_mel_collate, drop_last=True)

    def val_dataloader(self):
        valset = TextMelDataset(self.hp, self.hp.data.val_dir, self.hp.data
            .val_meta, train=False, norm=True)
        return paddle.io.DataLoader(valset, batch_size=self.hp.train
            .batch_size, shuffle=False, num_workers=self.hp.train.
            num_workers, collate_fn=text_mel_collate, drop_last=False)

    def load_checkpoint(self, checkpoint_path):
        if checkpoint_path is None:
            return 0, None
        if os.path.isfile(checkpoint_path):
            checkpoint = paddle.load(path=checkpoint_path)
            if 'state_dict' in checkpoint:
                self.model.set_state_dict(state_dict=checkpoint['state_dict'])
            else:
                raise KeyError('No model state found in checkpoint file')
            epoch = checkpoint.get('epoch', 0)
            optimizer_state = checkpoint.get('optimizer', None)
            return epoch, optimizer_state
        else:
            return 0, None

    def train_and_val(self, checkpoint_path=None):
        cur_epoch, optimizer_state = self.load_checkpoint(checkpoint_path)
        trainloader = self.train_dataloader()
        valloader = self.val_dataloader()
        optimizer = self.configure_optimizers()
        if optimizer_state:
            optimizer.set_state_dict(state_dict=optimizer_state)
        self.model.load_cotatron(self.hp.train.cotatron_path)
        for epoch in range(cur_epoch, self.hp.train.max_epochs):
            self.model.train()
            train_loss = 0.0
            train_progress_bar = tqdm(trainloader, desc=
                f'Epoch {epoch + 1}/{self.hp.train.max_epochs}')
            for batch in train_progress_bar:
                optimizer.clear_grad()
                loss = self.training_step(batch)['loss']
                train_loss += loss.item()
                loss.backward()
                self.global_step += 1
                optimizer.step()
                train_progress_bar.set_postfix(loss_rec=loss.item())
            avg_train_loss = train_loss / len(trainloader)
            self.model.eval()
            val_loss = 0.0
            val_progress_bar = tqdm(valloader, desc=
                f'Epoch {epoch + 1}/{self.hp.train.max_epochs} evaluating')
            for batch in val_progress_bar:
                loss = self.validation_step(batch)['loss_rec']
                val_progress_bar.set_postfix(loss_rec=loss.item())
                val_loss += loss.item()
            avg_val_loss = val_loss / len(valloader)
            print(f'Average Validation Loss: {avg_val_loss:.4f}')
            epoch_log = {'epoch': epoch + 1, 'avg_train_loss':
                avg_train_loss, 'avg_val_loss': avg_val_loss}
            if os.path.exists(self.hp.log.log_dir) is False:
                os.makedirs(self.hp.log.log_dir)
            log_file_path = os.path.join(self.hp.log.log_dir, 'log.json')
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r+') as log_file:
                    log_data = json.load(log_file)
                    log_data.append(epoch_log)
                    log_file.seek(0)
                    json.dump(log_data, log_file, indent=4)
            else:
                with open(log_file_path, 'w') as log_file:
                    json.dump([epoch_log], log_file, indent=4)
            if os.path.exists(self.hp.log.ckpt_dir) is False:
                os.makedirs(self.hp.log.ckpt_dir)
            model_save_path = os.path.join(self.hp.log.ckpt_dir,
                f'model_epoch_{epoch + 1}_step{self.global_step}.pdparams')
            paddle.save(obj={'epoch': epoch + 1, 'state_dict': self.model.
                state_dict(), 'optimizer': optimizer.state_dict()}, path=
                model_save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', nargs=2, type=str, required=True,
        help='path of configuration yaml file')
    parser.add_argument('-g', '--gpus', type=str, default=None, help=
        "Number of gpus to use (e.g. '0,1,2,3'). Will use all if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
        help='path of checkpoint for resuming')
    parser.add_argument('-s', '--save_top_k', type=int, default=-1, help=
        'save top k checkpoints, default(-1): save all')
    parser.add_argument('-f', '--fast_dev_run', type=bool, default=False,
        help='fast run for debugging purpose')
    parser.add_argument('--val_epoch', type=int, default=1, help=
        'run val loop every * training epochs')
    args = parser.parse_args()
    paddle.set_device('gpu')
    model = Synthesizer(args)
    trainer = MySynthesizerTrainer(args, model)
    trainer.train_and_val()
