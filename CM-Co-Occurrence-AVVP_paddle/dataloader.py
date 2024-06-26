import paddle
import os
import numpy as np
import pandas as pd
from ipdb import set_trace
import pickle as pkl


def ids_to_multinomial(ids):
    """ label encoding

    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat',
        'Frying_(food)', 'Basketball_bounce', 'Fire_alarm', 'Chainsaw',
        'Cello', 'Banjo', 'Singing', 'Chicken_rooster', 'Violin_fiddle',
        'Vacuum_cleaner', 'Baby_laughter', 'Accordion', 'Lawn_mower',
        'Motorcycle', 'Helicopter', 'Acoustic_guitar',
        'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender', 'Clapping']
    id_to_idx = {id: index for index, id in enumerate(categories)}
    y = np.zeros(len(categories))
    for id in ids:
        index = id_to_idx[id]
        y[index] = 1
    return y


class LLP_dataset(paddle.io.Dataset):

    def __init__(self, label, audio_dir, video_dir, st_dir, transform=None):
        self.df = pd.read_csv(label, header=0, sep='\t')
        self.filenames = self.df['filename']
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.st_dir = st_dir
        self.transform = transform
        self.need_to_change_v, self.need_to_change_a = pkl.load(open(
            'need_to_change.pkl', 'rb'))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # row = self.df.loc[idx, :]
        row = self.df.iloc[idx]  
        name = row.iloc[0][:11]  
        audio = np.load(os.path.join(self.audio_dir, name + '.npy'))
        video_s = np.load(os.path.join(self.video_dir, name + '.npy'))
        video_st = np.load(os.path.join(self.st_dir, name + '.npy'))
        # ids = row[-1].split(',')
        ids = row.iloc[-1].split(',') 
        label = ids_to_multinomial(ids)
        sample = {'audio': audio, 'video_s': video_s, 'video_st': video_st,
            'label': label}
        if self.transform:
            sample = self.transform(sample)
        pa = label.copy()
        pv = label.copy()
        for c in range(25):
            if label[c] != 0:
                if idx in self.need_to_change_v[c]:
                    pv[c] = 0
                if idx in self.need_to_change_a[c]:
                    pa[c] = 0
        sample['v_refine'] = pv
        sample['a_refine'] = pa
        sample['name'] = self.filenames[idx]
        return sample


class ToTensor(object):

    def __call__(self, sample):
        if len(sample) == 2:
            audio = sample['audio']
            label = sample['label']
            return {'audio': paddle.to_tensor(data=audio), 'label': paddle.
                to_tensor(data=label)}
        else:
            audio = sample['audio']
            video_s = sample['video_s']
            video_st = sample['video_st']
            label = sample['label']
            return {'audio': paddle.to_tensor(data=audio), 'video_s':
                paddle.to_tensor(data=video_s), 'video_st': paddle.
                to_tensor(data=video_st), 'label': paddle.to_tensor(data=label)
                }
