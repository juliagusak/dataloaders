from torch.utils import data
import glob
import librosa
import numpy as np
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
#from .transforms import get_train_transform, get_test_transform
from sklearn import preprocessing
from .utils import *


## TO DO : Transfroms

class GTZAN(data.Dataset):
    def __init__(self,
                 root_dir=None,sr = 16000,
                 precision = np.float32,
                 seed = 42,
                 transform=None,verbose=0):
        if root_dir is None:
            root_dir = '/data/datasets/audio/genres/'
        self.root_dir = root_dir
        self.file_list = sorted(glob.glob(root_dir + '/**/*.*'))
        self.X = []
        self.y = []
        self.seed = seed
        self.sr = sr
        self.transform = transform
        self.le = preprocessing.LabelEncoder()
        self.precision = precision
        
        ## reduce reading from 10 minutes to 42 seconds
        if verbose:
            iterable = tqdm(self.file_list)
        else:
            iterable = self.file_list
        parres = Parallel(n_jobs=-1, verbose=0)(delayed(self.__reader__)(f) for f in iterable) 
        for wave,label in parres:
            self.X.append(wave)
            self.y.append(label)
        self.le.fit(self.y)
        self.n_classes = len(self.le.classes_)
        
        assert len(self.X) == len(self.y)
        
    def __len__(self):
        return len(self.X)
    
    def __reader__(self,f):
        label = f.split('/')[-2]
        wave, sr = librosa.core.load(f,self.sr,res_type='kaiser_fast')
        return (wave,label)
    
    def __do_transform(self, x, y):
        y_enc = self.le.transform([y])[0]
        y = numpy_one_hot(y_enc, num_classes=self.n_classes)
        x = x.astype(self.precision)
        sample = {IMG: x, LABEL: y}
        if self.transform:
            pass
        return sample
    
    def __getitem__(self, index):
        return self.__do_transform(self.X[index],self.y[index])