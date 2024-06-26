# Assem-VC &mdash; Paddlepaddle Implementation


## Requirements

This repository was tested with following environment:

- Python 3.8
- paddlepaddle
- The requirements are highlighted in [requirements.txt](./requirements.txt).



## Datasets

### Preparing Data

- To reproduce the results from our paper, you need to download:
  - LibriTTS train-clean-100 split [tar.gz link](http://www.openslr.org/resources/60/train-clean-100.tar.gz)
  - [VCTK dataset (Version 0.80)](https://datashare.ed.ac.uk/handle/10283/2651)
- Unzip each files, and clone them in `datasets/`.
- Resample them into 22.05kHz using `datasets/resample.py`.
  ```bash
  python datasets/resample.py
  ```
  Note that `dataset/resample.py` was hard-coded to remove original wavfiles in `datasets/` and replace them into resampled wavfiles,
  and their filename `*.wav` will be transformed into `*-22k.wav`.
- You can use `datasets/resample_delete.sh` instead of `datasets/resample.py`. It does the same role.


### Preparing Metadata

Following a format from [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2), the metadata should be formatted like:
```
path_to_wav|transcription|speaker_id
path_to_wav|transcription|speaker_id
...
```

When you want to learn and inference using phoneme, the transcription should have only unstressed [ARPABET](https://en.wikipedia.org/wiki/ARPABET).

Metadata containing ARPABET for LibriTTS train-clean-100 split and VCTK corpus are already prepared at `datasets/metadata`.
If you wish to use custom data, you need to make the metadata as shown above.

When converting transcription of metadata into ARPABET, you can use `datasets/g2p.py`.

```bash
python datasets/g2p.py -i <input_metadata_filename_with_graphemes> -o <output_filename>
```

### Preparing Configuration Files

Training our VC system is consisted of two steps: (1) training Cotatron, (2) training VC decoder on top of Cotatron.

There are three `yaml` files in the `config` folder, which are configuration template for each model.
They **must** be edited to match your training requirements (dataset, metadata, etc.).

```bash
cp config/global/default.yaml config/global/config.yaml
cp config/cota/default.yaml config/cota/config.yaml
cp config/vc/default.yaml config/vc/config.yaml
```

Here, all files with name other than `default.yaml` will be ignored from git (see `.gitignore`).

- `config/global`: Global configs that are both used for training Cotatron & VC decoder.
  - Fill in the blanks of: `speakers`, `train_dir`, `train_meta`, `val_dir`, `val_meta`, `f0s_list_path`.
  - Example of speaker id list is shown in `datasets/metadata/libritts_vctk_speaker_list.txt`.
  - When replicating the two-stage training process from our paper (training with LibriTTS and then LibriTTS+VCTK), please put both list of speaker ids from LibriTTS and VCTK at global config.
  - `f0s_list_path` is set to `f0s.txt` by default
- `config/cota`: Configs for training Cotatron.
  - You may want to change: `batch_size` for GPUs other than 32GB V100, or change `chkpt_dir` to save checkpoints in other disk.
  - You can also modify `use_attn_loss`, whether guided attention loss is used or not.
- `config/vc`: Configs for training VC decoder.
  - Fill in the blank of: `cotatron_path`. 

### Extracting Pitch Range of Speakers

Before you train VC decoder, you should extract pitch range of each speaker:

```bash
python preprocess.py -c <path_to_global_config_yaml>
```
Result will be saved at `f0s.txt`.

## Training
### 1. Training Cotatron
To train the Cotatron, run this command:

```bash
python my_cotatron_trainer.py -c <path_to_global_config_yaml> <path_to_cotatron_config_yaml> -g <gpus>
```

Here are some example commands that might help you understand the arguments:

```bash

python my_cotatron_trainer.py -c config/global/config.yaml config/cota/config.yaml -g 0
```

Optionally, you can resume the training from previously saved checkpoint by adding `-p <checkpoint_path>` argument.

### 2. Training VC decoder

After the Cotatron is sufficiently trained (i.e., producing stable alignment + converged loss),
the VC decoder can be trained on top of it.

```bash
python my_synthesizer_trainer.py -c <path_to_global_config_yaml> <path_to_vc_config_yaml> -g <gpus>
```

The optional checkpoint argument is also available for VC decoder.


### Monitoring via Tensorboard

The progress of training with loss values and validation output can be monitored with tensorboard.
By default, the logs will be stored at `logs/cota` or `logs/vc`, which can be modified by editing `log.log_dir` parameter at config yaml file.

```bash
tensorboard --log_dir logs/cota --bind_all # Cotatron - Scalars, Images, Hparams, Projector will be shown.
tensorboard --log_dir logs/vc --bind_all # VC decoder - Scalars, Images, Hparams will be shown.
```

## Pre-trained Weight
We provide pretrained model of Assem-VC and GTA-finetuned HiFi-GAN generator weight.
Assem-VC was trained with VCTK and LibriTTS, and HiFi-GAN was finetuned with VCTK.

1. Download our published [models and configurations](https://drive.google.com/drive/folders/1aIl8ObHxsmsFLXBz-y05jMBN4LrpQejm?usp=sharing).
2. Place `global/config.yaml` at `config/global/config.yaml`, and `vc/config.yaml` at `config/vc/config.yaml`
3. Download `f0s.txt` and write the relative path of it at `hp.data.f0s_list_path`.
(Default path is `f0s.txt`)
4. write path of pretrained Assem-VC and HiFi-GAN models in [inference.py](./inference.py).

## Inference

After the VC decoder are trained, you can use an arbitrary speaker's speech as the source.
You can convert it to speaker contained in trainset: which is any-to-many voice conversion.
1. Add your source audio(.wav) in `datasets/inference_source`
2. Add following lines at `datasets/inference_source/metadata_origin.txt`
    ```
    your_audio.wav|transcription|speaker_id
    ```
    Note that speaker_id has no effect whether or not it is in the training set.
3. Convert `datasets/inference_source/metadata_origin.txt` into ARPABET.
    ```bash
    python datasets/g2p.py -i datasets/inference_source/metadata_origin.txt \
                            -o datasets/inference_source/metadata_g2p.txt
    ```
4. Run [inference.py](./inference.py)

