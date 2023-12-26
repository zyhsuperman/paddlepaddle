# One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization

# Dependencies
- python 3.6+
- paddlepaddle
- numpy 1.16.0
- librosa 0.6.3
- SoundFile 0.10.2 
- tensorboardX


# Preprocess
We provide the preprocess script for two datasets: VCTK and LibriTTS. The download links are below.
- [CSTR VCTK Corpus](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)
- [LibriTTS Corpus](http://www.openslr.org/60/)

The experiments in the paper is done on VCTK. 

The preprocess code is at ```preprocess/```.
The configuation for preprocessing is at ```preprocess/libri.config``` and ```preprocess/vctk.config```. Depends on which dataset you used.
where:
- **segment\_size** is the segment size for training. Default: 128
- **data\_dir** is the directory to put preprocessed files. 
- **raw\_data\_dir** is the directory to put the raw data. Like ```LibriTTS/``` or ```VCTK-Corpus/```.
- **n_out_speakers** is the number of speakers for testing. Default: 20.
- **test\_prop** is the proportion for validation utterances. Default: 0.1
- **training\_samples** is the number of sampled segments for training (we sample it in the preprocess stage). Default: 10000000.
- **testing_samples** is the number of sampled segments for testing. Default: 10000.
- **n\_utt\_attr** is the number of utterances to compute mean and standard deviation for normalization. Default: 5000.
- **train_set**: only for LibriTTS. The subset used for training. Default: train-clean-100.
- **test_set**: only for LibriTTS. The subset used for testing. Default: dev-clean.

Once you edited the config file, you can run ```preprocess_vctk.sh``` or ```preprocess_libri.sh``` to preprocess the dataset. 
<br>
Also, you can change the feature extraction config in ```preprocess/tacotron/hyperparams.py```

# Training
The default arguments can be found in ```train.sh```. The usage of each arguments are listed below. 
- **-c**: the path of config file, the default hyper-parameters can be found at ```config.yaml```.
- **-iters**: train the model with how many iterations. default: 200000
- **-summary_steps**: record training loss every n steps.
- **-t**: the tag for tensorboard.
- **-train_set**: the data file for training (```train``` if the file is train.pkl). Default: ```train```
- **-train_index_file**: the name of training index file. Default: ```train_samples_128.json```
- **-data_dir**: the directory for processed data.
- **-store_model_path**: the path to store the model.

# Inference
You can use ```inference.py``` to inference.
- **-c**: the path of config file.
- **-m**: the path of model checkpoint.
- **-a**: the attribute file for normalization ad denormalization.
- **-s**: the path of source file (.wav).
- **-t**: the path of target file (.wav).
- **-o**: the path of output converted file (.wav).

