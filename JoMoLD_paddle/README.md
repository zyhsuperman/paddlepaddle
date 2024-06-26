## Joint-Modal Label Denoising for Weakly-Supervised Audio-Visual Video Parsing

Paddlepaddle Implementation ECCV 2022 paper [Joint-Modal Label Denoising for Weakly-Supervised Audio-Visual Video Parsing](https://arxiv.org/abs/2204.11573)


## Get Started


### Prepare data

1. Please download the preprocessed audio and visual features from https://github.com/YapengTian/AVVP-ECCV20.
2. Put the downloaded features into data/feats/.


### Train the model

1.Train noise estimator:
```bash
python main.py --mode train_noise_estimator --save_model true --model_save_dir ckpt --checkpoint noise_estimater.pdparams
```
2.Calculate noise ratios:
```bash
python main.py --mode calculate_noise_ratio --model_save_dir ckpt --checkpoint noise_estimater.pdparams --noise_ratio_file noise_ratios.npz
```
3.Train model with label denoising:
```bash
python main.py --mode train_label_denoising --save_model true --model_save_dir ckpt --checkpoint JoMoLD.pdparams --noise_ratio_file noise_ratios.npz
```

### Test

Please download and put the checkpoint into "./ckpt" directory and use the following command to test:
```bash
python main.py --mode test_JoMoLD --model_save_dir ckpt --checkpoint JoMoLD.pdparams
```

