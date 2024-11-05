
### Dependencies
- Python 3.8
- Numpy
- paddlepaddle
- librosa
- tqdm



### 0.Convert Mel-Spectrograms

run ```python conversion.py```


### 1.Mel-Spectrograms to waveform

Download pre-trained WaveNet Vocoder model, and run the ```python vocoder.py``` in the same the directory.

Please note the training metadata and testing metadata have different formats.


### 2.Train model

We have included a small set of training audio files in the wav folder. However, the data is very small and is for code verification purpose only. Please prepare your own dataset for training.

1.Generate spectrogram data from the wav files: ```python make_spect.py```

2.Generate training metadata, including the GE2E speaker embedding (please use one-hot embeddings if you are not doing zero-shot conversion): ```python make_metadata.py```

3.Run the main training script: ```python main.py```

Converges when the reconstruction loss is around 0.0001.



