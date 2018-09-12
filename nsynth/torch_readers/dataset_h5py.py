import h5py

import numpy as np

from mics.basic_dataset import BasicDataset
from mics.utils import LabelsEncoder, LabelsToOneHot
from nsynth.constants import *
from nsynth.torch.basic_dataset import NSynthBasicDataset


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
        super(NSynthBasicDataset, self).__init__(dataset_path,
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

    def read_file(self, dataset_path):
        print("!!!!!!!!")
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

    def instance_dataset(self, dataset_path, transforms, in_memory):
        new_dataset = NSynthH5PyDataset(dataset_path,
                                        transforms,
                                        sr=self.sr,
                                        signal_length=self.signal_length,
                                        precision=self.precision,
                                        one_hot_all=False,
                                        one_hot_pitch=False,
                                        one_hot_velocity=False,
                                        one_hot_instr_src=False,
                                        one_hot_instr_family=False,
                                        encode_cat=False,
                                        in_memory=in_memory
                                        )

        new_dataset.one_hot_all = self.one_hot_all
        if self.one_hot_pitch or self.one_hot_all:
            new_dataset.one_hot_pitch = True
            new_dataset.pitch_one_hot = self.pitch_one_hot

        if self.one_hot_velocity or self.one_hot_all:
            new_dataset.one_hot_velocity = True
            new_dataset.velocity_one_hot = self.velocity_one_hot

        if self.one_hot_instr_src or self.one_hot_all:
            new_dataset.one_hot_instr_src = True
            new_dataset.instr_src_one_hot = self.instr_src_one_hot

        if self.one_hot_instr_family or self.one_hot_all:
            new_dataset.one_hot_instr_family = True
            new_dataset.instr_fml_one_hot = self.instr_fml_one_hot

        new_dataset.encode_cat = self.encode_cat
        if self.encode_cat:
            new_dataset.pitch_encoder = self.pitch_encoder
            new_dataset.velocity_encoder = self.velocity_encoder
            new_dataset.instr_src_encoder = self.instr_src_encoder
            new_dataset.instr_fml_encoder = self.instr_fml_encoder

            new_dataset.pitch = self.pitch_encoder(new_dataset.pitch)
            new_dataset.velocity = self.velocity_encoder(new_dataset.velocity)
            new_dataset.instr_src = self.instr_src_encoder(new_dataset.instr_src)
            new_dataset.instr_fml = self.instr_fml_encoder(new_dataset.instr_fml)

        return new_dataset

    def __exit__(self, exc_type, exc_value, traceback):
        if self.hpy_file is not None:
            self.hpy_file.close()

    def __getitem__(self, index):
        audio, pitch, velocity = self.audio[index], self.pitch[index], self.velocity[index]
        instrument_source, instrument_family, = self.instr_src[index], self.instr_fml[index]
        qualities = self.qualities[index]

        audio = self.do_transform(audio)

        if self.one_hot_pitch or self.one_hot_all:
            pitch = self.do_one_hot(pitch, self.pitch_one_hot)
        if self.one_hot_velocity or self.one_hot_all:
            velocity = self.do_one_hot(velocity, self.velocity_one_hot)
        if self.one_hot_instr_src or self.one_hot_all:
            instrument_source = self.do_one_hot(instrument_source, self.instr_src_one_hot)
        if self.one_hot_instr_family or self.one_hot_all:
            instrument_family = self.do_one_hot(instrument_family, self.instr_fml_one_hot)

        return {AUDIO: audio, PITCH: pitch, VELOCITY: velocity,
                INSTR_SRC: instrument_source, INSTR_FAMILY: instrument_family, QUALITIES: qualities}


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

    dataset = dataset.instance_dataset("../nsynth-test.hdf5", train_transforms, True)

    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 1}
    training_generator = data.DataLoader(dataset, **params)

    for batch in training_generator:
        print(batch[AUDIO].shape)
        print(batch[PITCH])
        break
