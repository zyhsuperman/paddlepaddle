import paddle
"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
paddle.set_device('gpu')
C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256)
C.eval()
c_checkpoint = paddle.load(path='checkpoints/speaker_encoder.pdparams')
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.set_state_dict(state_dict=new_state_dict)
num_uttrs = 10
len_crop = 128
rootDir = './spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)
speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))
    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    for i in range(num_uttrs):
        tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
        candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
        while tmp.shape[0] < len_crop:
            idx_alt = np.random.choice(candidates)
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
            candidates = np.delete(candidates, np.argwhere(candidates ==
                idx_alt))
        left = np.random.randint(0, tmp.shape[0] - len_crop)
        melsp = paddle.to_tensor(data=tmp[(np.newaxis), left:left +
            len_crop, :])
        emb = C(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())
    utterances.append(np.mean(embs, axis=0))
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker, fileName))
    speakers.append(utterances)
with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)
