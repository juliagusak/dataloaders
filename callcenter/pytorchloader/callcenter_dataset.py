import numpy as np
import pandas as pd
import librosa

from torch.utils import data

from misc.transforms import get_train_transform, get_test_transform
from misc.utils import FEATURES, LABEL, numpy_one_hot, mix, tensor_to_numpy





class CallCenterDataset(data.Dataset):
    def __init__(self, data_path,
                 csv_local_path,
                 sr =8000, 
                 is_train=True,
                 signal_length=2**16,
                 mix=False,
                 precision=np.float32,
                 n_files = None,
                 upsample_factor = 1):

        self.signal_length = signal_length

        if is_train:
            self.transform = get_train_transform(length=signal_length)
        else:
            self.transform = get_test_transform(length=signal_length)

        self.sr = sr
        self.mix = mix
        self.precision = precision
        self.n_files = n_files
        self.upsample_factor = upsample_factor
        
        df = pd.read_csv(data_path + csv_local_path)
        
        if self.n_files is not None:
            df = df.sample(self.n_files)
        
        df['file_name'] = df['file_name'].apply(lambda x: '{}/{}'.format(data_path,
                                               '/'.join(['callCenterDataset',
                                                         x.split('callCenterDataset/')[1]])))
        
        self.X = []
        self.y = []
        
        for idx, row in df.iterrows():          
            v_start,v_end = row['v_start'],row['v_end']
            
            signal, sr = librosa.load(file_name, sr = self.sr,
                                      res_type = 'kaiser_fast')
            assert (len(signal[int(v_start*sr):int(v_end*sr)]) > 0)
            
            self.X.append(signal[int(v_start*sr):int(v_end*sr)])
            self.y.append(int(row['is_human']))
        
        self.n_classes = len(set(self.y))
        
        
    def __len__(self):
        '''
        Denotes the total number of samples
        '''
        return len(self.y)
        
    def __do_transform(self, signal):
        signal = signal.astype(self.precision)
        if self.transform:
            signal = tensor_to_numpy(self.transform(signal.reshape((1, -1, 1))))
            
            signal = np.repeat(signal, repeats=self.upsample_factor, axis = -1)

        return signal

    def __mix_samples(self, sample1, sample2):
        r = np.random.uniform()

        sound1 = sample1[FEATURES].reshape((-1))
        sound2 = sample2[FEATURES].reshape((-1))
        
        sound = mix(sound1, sound2, r, self.sr*self.upsample_factor)
        label = r * sample1[LABEL] + (1.0 - r) * sample2[LABEL]
        
        sound = sound.reshape((1, 1, -1))

        return {FEATURES: sound, LABEL: label}

    def __getitem__(self, index):
        if self.mix:
            idx1, idx2 = np.random.choice(len(self), 2, replace=False)

            sample1 = {FEATURES: self.__do_transform(self.X[idx1]),
                       LABEL: numpy_one_hot(self.y[idx1], num_classes=self.n_classes)}
            sample2 = {FEATURES: self.__do_transform(self.X[idx2]),
                       LABEL: numpy_one_hot(self.y[idx2], num_classes=self.n_classes)}

            sample = self.__mix_samples(sample1, sample2)

        else:
            sample = {FEATURES: self.__do_transform(self.X[index]),
                      LABEL: numpy_one_hot(self.y[index], num_classes=self.n_classes)}

        return sample