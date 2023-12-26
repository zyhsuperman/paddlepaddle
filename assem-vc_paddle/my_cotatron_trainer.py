import paddle
import os
import random
import numpy as np
from omegaconf import OmegaConf
from cotatron import Cotatron
from argparse import ArgumentParser
from datasets import TextMelDataset, text_mel_collate
from tqdm import tqdm
import json
import os
from datasets.text import Language


class CotatronTrainer:

    def __init__(self, hparams, model):
        super().__init__()
        self.hparams = hparams
        hp_global = OmegaConf.load(hparams.config[0])
        hp_cota = OmegaConf.load(hparams.config[1])
        hp = OmegaConf.merge(hp_global, hp_cota)
        self.hp = hp
        self.model = model
        self.global_step = 0
        self.symbols = Language(hp.data.lang, hp.data.text_cleaners
            ).get_symbols()
        self.symbols = ['"{}"'.format(symbol) for symbol in self.symbols]

    def training_step(self, batch):
        (text, mel_target, speakers, input_lengths, output_lengths,
            max_input_len, _) = batch
        speaker_emb, mel_pred, mel_postnet, alignment = self.model(text,
            mel_target, speakers, input_lengths, output_lengths, max_input_len)
        speaker_prob = self.model.classifier(speaker_emb)
        classifier_loss = paddle.nn.functional.nll_loss(input=speaker_prob,
            label=speakers)
        loss_rec = paddle.nn.functional.mse_loss(input=mel_pred, label=
            mel_target) + paddle.nn.functional.mse_loss(input=mel_postnet,
            label=mel_target)
        if self.model.use_attn_loss:
            attention_loss = self.model.attn_loss(alignment, input_lengths,
                output_lengths, self.global_step)
            return {'total_loss': loss_rec + classifier_loss +
                attention_loss, 'loss_rec': loss_rec, 'classifier_loss':
                classifier_loss, 'attention_loss': attention_loss}
        return {'total_loss': loss_rec + classifier_loss, 'loss_rec':
            loss_rec, 'classifier_loss': classifier_loss}

    def validation_step(self, batch):
        (text, mel_target, speakers, input_lengths, output_lengths,
            max_input_len, _) = batch
        speaker_emb, mel_pred, mel_postnet, alignment = self.model(text,
            mel_target, speakers, input_lengths, output_lengths,
            max_input_len, prenet_dropout=0.5, tfrate=False)
        speaker_prob = self.model.classifier(speaker_emb)
        classifier_loss = paddle.nn.functional.nll_loss(input=speaker_prob,
            label=speakers)
        loss_rec = paddle.nn.functional.mse_loss(input=mel_pred, label=
            mel_target) + paddle.nn.functional.mse_loss(input=mel_postnet,
            label=mel_target)
        return {'loss_rec': loss_rec, 'classifier_loss': classifier_loss}

    def configure_optimizers(self):
        return paddle.optimizer.Adam(parameters=self.model.parameters(),
            learning_rate=self.hp.train.adam.lr, weight_decay=self.hp.train
            .adam.weight_decay)

    def lr_lambda(self, step):
        progress = (step - self.hp.train.decay.start) / (self.hp.train.
            decay.end - self.hp.train.decay.start)
        return self.hp.train.decay.rate ** np.clip(progress, 0.0, 1.0)

    def optimizer_step(self, optimizer):
        lr_scale = self.lr_lambda(self.global_step)

        optimizer.set_lr(lr_scale * self.hp.train.adam.lr)
        optimizer.step()

    def train_dataloader(self):
        trainset = TextMelDataset(self.hp, self.hp.data.train_dir, self.hp.
            data.train_meta, train=True, norm=False, use_f0s=False)
        return paddle.io.DataLoader(trainset, batch_size=self.hp.
            train.batch_size, shuffle=True, num_workers=self.hp.train.
            num_workers, collate_fn=text_mel_collate, drop_last=True)

    def val_dataloader(self):
        valset = TextMelDataset(self.hp, self.hp.data.val_dir, self.hp.data
            .val_meta, train=False, norm=False, use_f0s=False)
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

    def clip_grad_value(self, parameters, clip_value, norm_type=2):
        if isinstance(parameters, paddle.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        norm_type = float(norm_type)
        clip_value = float(clip_value)

        total_norm = 0
        for p in parameters:
            # 使用 PaddlePaddle 的 norm 方法计算梯度的范数
            param_norm = paddle.norm(p.grad, p=norm_type)
            total_norm += param_norm.item() ** norm_type

            # 使用 paddle.clip 裁剪梯度，并更新梯度值
            clipped_grad = paddle.clip(p.grad, min=-clip_value, max=clip_value)
            p.grad.set_value(clipped_grad)
        total_norm = total_norm ** (1. / norm_type)
        return total_norm

    def train_and_val(self, checkpoint_path=None):
        cur_epoch, optimizer_state = self.load_checkpoint(checkpoint_path)
        trainloader = self.train_dataloader()
        valloader = self.val_dataloader()
        optimizer = self.configure_optimizers()
        if optimizer_state:
            optimizer.set_state_dict(state_dict=optimizer_state)
        for epoch in range(cur_epoch, self.hp.train.max_epochs):
            self.model.train()
            train_loss = 0.0
            train_progress_bar = tqdm(trainloader, desc=
                f'Epoch {epoch + 1}/{self.hp.train.max_epochs}')
            for batch in train_progress_bar:

                optimizer.clear_grad()
                losses = self.training_step(batch)
                total_loss = losses['total_loss']
                train_loss += total_loss.item()
                total_loss.backward()
                self.global_step += 1
                if self.hp.train.grad_clip > 0.0:
                    grad_norm = self.clip_grad_value(self.model.parameters(
                        ), self.hp.train.grad_clip)
                self.optimizer_step(optimizer)
                if len(losses) == 4:
                    train_progress_bar.set_postfix(loss_rec=losses[
                        'loss_rec'].item(), classifier_loss=losses[
                        'classifier_loss'].item(), attention_loss=losses[
                        'attention_loss'].item())
                else:
                    train_progress_bar.set_postfix(loss_rec=losses[
                        'loss_rec'].item(), classifier_loss=losses[
                        'classifier_loss'].item())
            avg_train_loss = train_loss / len(trainloader)
            self.model.eval()
            val_loss = []
            val_progress_bar = tqdm(valloader, desc=
                f'Epoch {epoch + 1}/{self.hp.train.max_epochs} evaluating')
            for batch in val_progress_bar:
                losses = self.validation_step(batch)
                val_loss.append(losses['loss_rec'].item() + losses[
                    'classifier_loss'].item())
                val_progress_bar.set_postfix(loss_rec=losses['loss_rec'].
                    item(), classifier_loss=losses['classifier_loss'].item())
            avg_val_loss = sum(val_loss) / len(val_loss)
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
            if not os.path.exists(self.hp.log.ckpt_dir):
                os.makedirs(self.hp.log.ckpt_dir)
            model_save_path = os.path.join(self.hp.log.ckpt_dir,
                f'model_epoch_{epoch + 1}_step_{self.global_step}.pdparams')
            paddle.save(obj={'epoch': epoch + 1, 'state_dict': self.model.
                state_dict(), 'optimizer': optimizer.state_dict()}, path=
                model_save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, nargs=2, required=True,
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
    model = Cotatron(args)
    trainer = CotatronTrainer(args, model)
    trainer.train_and_val()
