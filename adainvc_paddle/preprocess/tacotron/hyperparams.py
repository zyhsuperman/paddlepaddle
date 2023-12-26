"""
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
"""


class Hyperparams:
    """Hyper parameters"""
    prepro = False
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"
    data = '/data/private/voice/LJSpeech-1.0'
    test_data = 'harvard_sentences.txt'
    max_duration = 10.0
    top_db = 15
    sr = 24000
    n_fft = 2048
    frame_shift = 0.0125
    frame_length = 0.05
    hop_length = int(sr * frame_shift)
    win_length = int(sr * frame_length)
    n_mels = 512
    power = 1.2
    n_iter = 100
    preemphasis = 0.97
    max_db = 100
    ref_db = 20
    embed_size = 256
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5
    dropout_rate = 0.5
    lr = 0.001
    logdir = 'logdir/01'
    sampledir = 'samples'
    batch_size = 32
