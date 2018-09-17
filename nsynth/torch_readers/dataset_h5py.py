import h5py

import numpy as np

from nsynth.constants import *
from nsynth.torch_readers.basic_dataset import NSynthBasicDataset


class NSynthH5PyDataset(NSynthBasicDataset):
    def __init__(self,
                 dataset_path,
                 transforms,
                 sr,
                 signal_length=2 ** 16,
                 precision=np.float32,
                 one_hot_all=False,
                 one_hot_pitch=False,
                 one_hot_velocity=False,
                 one_hot_instr_src=False,
                 one_hot_instr_family=False,
                 encode_cat=False,
                 in_memory=True):
        super(NSynthH5PyDataset, self).__init__(dataset_path,
                                                transforms,
                                                sr,
                                                signal_length=signal_length,
                                                precision=precision,
                                                one_hot_all=one_hot_all,
                                                one_hot_pitch=one_hot_pitch,
                                                one_hot_velocity=one_hot_velocity,
                                                one_hot_instr_src=one_hot_instr_src,
                                                one_hot_instr_family=one_hot_instr_family,
                                                encode_cat=encode_cat,
                                                in_memory=in_memory)
        self.hpy_file = None

    def read_file(self, dataset_path):
        f = h5py.File(dataset_path, 'r')
        self.pitch = f[PITCH][:]
        self.velocity = f[VELOCITY][:]
        self.instr_src = f[INSTR_SRC][:]
        self.instr_fml = f[INSTR_FAMILY][:]
        self.qualities = f[QUALITIES][:]
        if self.in_memory:
            self.audio = f[AUDIO][:]
            f.close()
        else:
            self.hpy_file = f
            self.audio = f[AUDIO]

    def __exit__(self, exc_type, exc_value, traceback):
        if self.hpy_file is not None:
            self.hpy_file.close()

    def read_elem(self, index):
        audio, pitch, velocity = self.audio[index], self.pitch[index], self.velocity[index]
        instrument_source, instrument_family = self.instr_src[index], self.instr_fml[index]
        qualities = self.qualities[index]

        return audio, pitch, velocity, instrument_source, instrument_family, qualities


if __name__ == "__main__":
    from mics.transforms import get_train_transform
    from torch.utils import data

    train_transforms = get_train_transform(length=2 ** 14)
    dataset = NSynthH5PyDataset("../nsynth-test.hdf5",
                                one_hot_pitch=True,
                                encode_cat=True,
                                transforms=train_transforms,
                                sr=16000,
                                in_memory=True)
    print("Dataset Len", len(dataset))
    print("item 0", dataset[0])

    dataset = dataset.instance_dataset("../nsynth-test.hdf5", train_transforms, False)

    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 1}
    training_generator = data.DataLoader(dataset, **params)

    for batch in training_generator:
        print(batch[AUDIO].shape)
        print(batch[PITCH])
        break
