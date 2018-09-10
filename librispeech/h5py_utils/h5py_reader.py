import h5py

import numpy as np

from torch.utils import data

from librispeech.basic_reader import LibriSpeechBasic
from mics.utils import LabelsToOneHot
from librispeech.h5py_utils.utils import SPEAKER, SOUND, CHAPTER, UTTERANCE


class LibriSpeechH5py(LibriSpeechBasic):
    def __init__(self,
                 dataset_path,
                 transforms,
                 sr,
                 signal_length=2 ** 16,
                 precision=np.float32,
                 one_hot_all=False,
                 one_hot_speaker=False,
                 one_hot_chapter=False,
                 one_hot_utterance=False,
                 in_memory=True):
        super(LibriSpeechH5py, self).__init__(transforms, sr, signal_length, precision, one_hot_all, in_memory)

        self.hpy_file = None
        f = h5py.File(dataset_path, 'r')
        self.speaker, self.chapter, self.utterance = f[SPEAKER][:], f[CHAPTER][:], f[UTTERANCE][:]
        if self.in_memory:
            self.sound = f[SOUND][:]
            f.close()
        else:
            self.hpy_file = f
            self.sound = f[SOUND]

        self.n = self.speaker.shape[0]

        self.one_hot_speaker = one_hot_speaker
        self.one_hot_chapter = one_hot_chapter
        self.one_hot_utterance = one_hot_utterance

        if self.one_hot_speaker or self.one_hot_all:
            self.label_encoder = LabelsToOneHot(self.speaker)
        else:
            self.label_encoder = None

        if self.one_hot_chapter or self.one_hot_all:
            self.chapter_encoder = LabelsToOneHot(self.chapter)
        else:
            self.chapter_encoder = None

        if self.one_hot_utterance or self.one_hot_all:
            self.utterance_encoder = LabelsToOneHot(self.utterance)
        else:
            self.utterance_encoder = None

    def __exit__(self, exc_type, exc_value, traceback):
        if self.hpy_file is not None:
            self.hpy_file.close()

    def __getitem__(self, index):
        sound, speaker, chapter, utterance = self.sound[index], self.speaker[index], \
                                             self.chapter[index], self.utterance[index]
        sound = self.__do_transform(sound)

        if self.one_hot_speaker or self.one_hot_all:
            speaker = self.__do_one_hot(speaker, self.label_encoder)
        if self.one_hot_chapter or self.one_hot_all:
            chapter = self.__do_one_hot(chapter, self.chapter_encoder)
        if self.one_hot_utterance or self.one_hot_all:
            utterance = self.__do_one_hot(utterance, self.utterance_encoder)

        return {"sound": sound, "speaker": speaker, "chapter": chapter, 'utterance': utterance}


if __name__ == "__main__":
    from mics.transforms import get_train_transform

    train_transforms = get_train_transform(length=2 ** 14)
    dataset = LibriSpeechH5py("./librispeach/train-clean-100.hdf5",
                              transforms=train_transforms,
                              sr=16000,
                              one_hot_utterance=True,
                              in_memory=False)
    print("Dataset Len", len(dataset))
    print("item 0", dataset[0])

    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 1}
    training_generator = data.DataLoader(dataset, **params)

    for batch in training_generator:
        print(batch['sound'].shape)
        print(batch['utterance'])
        break
