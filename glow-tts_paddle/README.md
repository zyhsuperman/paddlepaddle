# Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search

## 1. Environments we use

* Python3.8
* paddlepaddle
* cython0.29.12
* librosa0.7.1
* numpy1.16.4
* scipy1.3.0

For Mixed-precision training, we use [apex](https://github.com/NVIDIA/apex); commit: 37cdaf4


## 2. Pre-requisites

a) Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/), then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY`

b) Build Monotonic Alignment Search Code (Cython): `cd monotonic_align; python setup.py build_ext --inplace`


## 3. Training Example

```sh
sh train_ddi.sh configs/base.json base
```

## 4. Inference Example

See [inference_hifigan.ipynb](./inference_hifigan.ipynb)

