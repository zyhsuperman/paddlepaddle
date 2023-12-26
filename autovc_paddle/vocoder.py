import paddle
import soundfile as sf
import pickle
from synthesis import build_model
from synthesis import wavegen

spect_vc = pickle.load(open('myresults.pkl', 'rb'))
paddle.set_device('gpu')
model = build_model()
checkpoint = paddle.load("autovc_paddle/checkpoints/wavnet.pdparams")
model.set_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(name)
    waveform = wavegen(model, c=c)   
    sf.write(name+'.wav', waveform, samplerate=16000)