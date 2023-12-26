# Modified HiFi-GAN for Assem-VC

## Pretrained Model
To GTA finetune HiFi-GAN models, you should download Pretrained models and transfer from those weight.

You can use pretrained `UNIVERSAL_V1` models authors of [HiFi-GAN](https://github.com/jik876/hifi-gan) provide.<br/> 
[Download pretrained models](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing) <br/> 
Details of each folder are as in follows:

|Folder Name|Generator|Dataset|Fine-Tuned|
|------|---|---|---|
|LJ_V1|V1|LJSpeech|No|
|LJ_V2|V2|LJSpeech|No|
|LJ_V3|V3|LJSpeech|No|
|LJ_FT_T2_V1|V1|LJSpeech|Yes ([Tacotron2](https://github.com/NVIDIA/tacotron2))|
|LJ_FT_T2_V2|V2|LJSpeech|Yes ([Tacotron2](https://github.com/NVIDIA/tacotron2))|
|LJ_FT_T2_V3|V3|LJSpeech|Yes ([Tacotron2](https://github.com/NVIDIA/tacotron2))|
|VCTK_V1|V1|VCTK|No|
|VCTK_V2|V2|VCTK|No|
|VCTK_V3|V3|VCTK|No|
|UNIVERSAL_V1|V1|Universal|No|

1. make cp_hifigan directory.
    ```bash
    mkdir cp_hifigan
    ```
2. Download `g_02500000` and `do_02500000` from [following link](https://drive.google.com/drive/folders/1YuOoV3lO2-Hhn1F2HJ2aQ4S0LC1JdKLd)
3. place them in `cp_hifigan/` directory.


## Fine-Tuning
1. Generate GTA mel-spectrograms in `torch.Tensor` format using [Assem-VC](https://github.com/mindslab-ai/assem-vc). <br/>
The file name of the generated mel-spectrogram should match the audio file and the extension should be `.gta`.<br/>
Example:
    ```
    Audio File : p233_392.wav
    Mel-Spectrogram File : p233_392.wav.gta
    ```
2. Run the following command.
    ```bash
    python train.py --config config_v1.json \
                    --input_wavs_dir <root_path_of_input_audios> \
                    --input_mels_dir <root_path_of_GTA_mels> \
                    --input_training_file <absolute_path_of_train_metadata_of_gta_mels> \
                    --input_validation_file <absolute_path_of_val_metadata_of_gta_mels> \
                    --fine_tuning True
    ```
    To train V2 or V3 Generator, replace `config_v1.json` with `config_v2.json` or `config_v3.json`.<br>
    Checkpoints and copy of the configuration file are saved in `cp_hifigan` directory by default.<br>
    You can change the path by adding `--checkpoint_path` option.
    
    Here are some example commands that might help you understand the arguments:

    ```bash
    python train.py --config config_v1.json \
                    --input_wavs_dir ../datasets/ \
                    --input_mels_dir ../datasets/ \
                    --input_training_file ../datasets/gta_metadata/gta_vctk_22k_train_10s_g2p.txt \
                    --input_validation_file ../datasets/gta_metadata/gta_vctk_22k_val_g2p.txt \
                    --fine_tuning True
    ```

### Monitoring via Tensorboard
```bash
tensorboard --log_dir cp_hifigan/logs --bind_all
```

## Acknowledgements
We referred to [HiFi-GAN](https://github.com/jik876/hifi-gan), [WaveGlow](https://github.com/NVIDIA/waveglow), [MelGAN](https://github.com/descriptinc/melgan-neurips) 
and [Tacotron2](https://github.com/NVIDIA/tacotron2) to implement this.

