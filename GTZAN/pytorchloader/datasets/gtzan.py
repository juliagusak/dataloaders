from torch.utils import data
import glob
import librosa
import numpy as np
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
from mics.transforms import get_train_transform, get_test_transform
from mics.utils import FEATURES, LABEL, numpy_one_hot, mix, tensor_to_numpy,LabelsToOneHot

class GTZAN(data.Dataset):
    def __init__(self,
                 root_dir,
                 sr = 16000,
                 precision = np.float32,
                 is_train = True,
                 seed = 42,
                 n_jobs = 8,
                 signal_length=2 ** 16,
                 verbose = 0):
        self.root_dir = root_dir
        self.file_list = sorted(glob.glob(self.root_dir + '/**/*.*'))
        self.X = []
        self.y = []
        self.is_train = is_train
        self.n_jobs = n_jobs
        self.seed = seed
        self.sr = sr
        self.precision = precision
        
        self.signal_length = signal_length
        if is_train:
            self.transform = get_train_transform(length=signal_length)
        else:
            self.transform = get_test_transform(length=signal_length)
        
        ## reduce reading from 10 minutes to 42 seconds
        if verbose:
            iterable = tqdm(self.file_list)
        else:
            iterable = self.file_list
        parres = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(self.__reader__)(f) for f in iterable) 
        for wave,label in parres:
            self.X.append(wave)
            self.y.append(label)
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.le = LabelsToOneHot(self.y)
        assert len(self.X) == len(self.y)
        
    def __len__(self):
        return len(self.X)
    
    def __reader__(self,f):
        label = f.split('/')[-2]
        wave, sr = librosa.core.load(f,self.sr,res_type='kaiser_fast')
        return (wave,label)
    
    def __do_transform(self, signal):
        signal = signal.astype(self.precision)
        if self.transform:
            signal = tensor_to_numpy(self.transform(signal.reshape((1, -1, 1))))
        return signal
    
    def __getitem__(self, index):
        y_enc = self.le(np.array(self.y[index]))[0]
        X_trans = self.__do_transform(self.X[index])
        sample = {FEATURES: X_trans , LABEL: y_enc}
        return sample