from abc import abstractmethod

import numpy as np

from misc.basic_dataset import BasicDataset
from misc.utils import LabelsEncoder, LabelsToOneHot
from nsynth.constants import *


class NSynthBasicDataset(BasicDataset):
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
        super(NSynthBasicDataset, self).__init__(transforms, sr, signal_length, precision,
                                                 one_hot_all, encode_cat, in_memory)
        self.one_hot_pitch = one_hot_pitch
        self.one_hot_velocity = one_hot_velocity
        self.one_hot_instr_src = one_hot_instr_src
        self.one_hot_instr_family = one_hot_instr_family

        self.audio = []
        self.pitch = []
        self.velocity = []
        self.instr_src = []
        self.instr_fml = []
        self.qualities = []

        self.read_file(dataset_path)

        self.n = self.pitch.shape[0]

        if self.encode_cat:
            self.pitch_encoder = LabelsEncoder(self.pitch)
            self.velocity_encoder = LabelsEncoder(self.velocity)
            self.instr_src_encoder = LabelsEncoder(self.instr_src)
            self.instr_fml_encoder = LabelsEncoder(self.instr_fml)

            self.pitch = self.pitch_encoder(self.pitch)
            self.velocity = self.velocity_encoder(self.velocity)
            self.instr_src = self.instr_src_encoder(self.instr_src)
            self.instr_fml = self.instr_fml_encoder(self.instr_fml)

        if self.one_hot_pitch or self.one_hot_all:
            self.pitch_one_hot = LabelsToOneHot(self.pitch)
        else:
            self.pitch_one_hot = None

        if self.one_hot_velocity or self.one_hot_all:
            self.velocity_one_hot = LabelsToOneHot(self.velocity)
        else:
            self.velocity_one_hot = None

        if self.one_hot_instr_src or self.one_hot_all:
            self.instr_src_one_hot = LabelsToOneHot(self.instr_src)
        else:
            self.instr_src_one_hot = None

        if self.one_hot_instr_family or self.one_hot_all:
            self.instr_fml_one_hot = LabelsToOneHot(self.instr_fml)
        else:
            self.instr_fml_one_hot = None

    @abstractmethod
    def read_file(self, dataset_path):
        pass

    @abstractmethod
    def read_elem(self, index):
        return None

    def instance_dataset(self, dataset_path, transforms, in_memory):
        new_dataset = self.__class__(dataset_path,
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

        return new_dataset

    def __getitem__(self, index):
        audio, pitch, velocity, instrument_source, instrument_family, qualities = self.read_elem(index)

        audio = self.do_transform(audio)

        if self.encode_cat and not self.in_memory:
            pitch = self.pitch_encoder(pitch)
            velocity = self.velocity_encoder(velocity)
            instrument_source = self.instr_src_encoder(instrument_source)
            instrument_family = self.instr_fml_encoder(instrument_family)

        if self.one_hot_pitch or self.one_hot_all:
            pitch = self.pitch_one_hot(pitch)
        if self.one_hot_velocity or self.one_hot_all:
            velocity = self.velocity_one_hot(velocity)
        if self.one_hot_instr_src or self.one_hot_all:
            instrument_source = self.instr_src_one_hot(instrument_source)
        if self.one_hot_instr_family or self.one_hot_all:
            instrument_family = self.instr_fml_one_hot(instrument_family)

        return {AUDIO: audio, PITCH: pitch, VELOCITY: velocity,
                INSTR_SRC: instrument_source, INSTR_FAMILY: instrument_family, QUALITIES: qualities}


if __name__ == "__main__":
    pass
