import numpy as np

from torch.utils import data

from mics.utils import tensor_to_numpy


class LibriSpeechBasic(data.Dataset):
    def __init__(self,
                 transforms,
                 sr,
                 signal_length=2 ** 16,
                 precision=np.float32,
                 one_hot_all=False,
                 in_memory=True):
        self.in_memory = in_memory
        self.transforms = transforms
        self.sr = sr
        self.signal_length = signal_length
        self.precision = precision

        self.n = None

        self.one_hot_all = one_hot_all

    def __do_transform(self, sound):
        if self.transforms:
            trans_sig = self.transforms(sound.reshape((1, -1, 1)))
            sound = tensor_to_numpy(trans_sig)

        return sound

    def __do_one_hot(self, id, encoder):
        return encoder(np.array([id]).reshape((-1, 1))).toarray()[0, :]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        pass
