import numpy as np

from torch.utils import data

from misc.utils import tensor_to_numpy


class BasicDataset(data.Dataset):
    def __init__(self,
                 transforms,
                 sr,
                 signal_length=2 ** 16,
                 precision=np.float32,
                 one_hot_all=False,
                 encode_cat=False,
                 in_memory=True):
        self.in_memory = in_memory
        self.transforms = transforms
        self.sr = sr
        self.signal_length = signal_length
        self.precision = precision

        self.n = None

        self.one_hot_all = one_hot_all
        self.encode_cat = encode_cat

    def do_transform(self, sound):
        if self.transforms:
            trans_sig = self.transforms(sound.reshape((1, -1, 1)))
            sound = tensor_to_numpy(trans_sig)

        return sound

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        pass
