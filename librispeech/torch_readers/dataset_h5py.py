import h5py

import numpy as np

from torch.utils import data

from mics.basic_dataset import BasicDataset
from mics.utils import LabelsToOneHot, LabelsEncoder
from librispeech.torch_readers.utils import SPEAKER, SOUND, CHAPTER, UTTERANCE


class H5PyDataset(BasicDataset):
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
                 encode_cat=False,
                 in_memory=True):
        super(H5PyDataset, self).__init__(transforms, sr, signal_length, precision,
                                          one_hot_all, encode_cat, in_memory)

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

        if self.encode_cat:
            self.speaker_encode = LabelsEncoder(self.speaker)
            self.chapter_encode = LabelsEncoder(self.chapter)
            self.utterance_encode = LabelsEncoder(self.utterance)

            self.speaker = self.speaker_encode(self.speaker)
            self.chapter = self.chapter_encode(self.chapter)
            self.utterance = self.utterance_encode(self.utterance)
        else:
            self.speaker_encode = None
            self.chapter_encode = None
            self.utterance_encode = None

        if self.one_hot_speaker or self.one_hot_all:
            self.speaker_one_hot = LabelsToOneHot(self.speaker)
        else:
            self.speaker_one_hot = None

        if self.one_hot_chapter or self.one_hot_all:
            self.chapter_one_hot = LabelsToOneHot(self.chapter)
        else:
            self.chapter_one_hot = None

        if self.one_hot_utterance or self.one_hot_all:
            self.utterance_one_hot = LabelsToOneHot(self.utterance)
        else:
            self.utterance_one_hot = None

    def instance_dataset(self, dataset_path, transforms, in_memory):
        new_dataset = H5PyDataset(dataset_path,
                                  transforms,
                                  sr=self.sr,
                                  signal_length=self.signal_length,
                                  precision=self.precision,
                                  one_hot_all=False,
                                  one_hot_speaker=False,
                                  one_hot_chapter=False,
                                  one_hot_utterance=False,
                                  encode_cat=False,
                                  in_memory=in_memory
                                  )

        new_dataset.one_hot_all = self.one_hot_all

        if self.one_hot_speaker or self.one_hot_all:
            new_dataset.one_hot_speaker = True
            new_dataset.pitch_one_hot = self.speaker_one_hot
        if self.one_hot_chapter or self.one_hot_all:
            new_dataset.one_hot_chapter = True
            new_dataset.chapter_one_hot = self.chapter_one_hot
        if self.one_hot_utterance or self.one_hot_all:
            new_dataset.one_hot_utterance = True
            new_dataset.utterance_one_hot = self.utterance_one_hot

        if self.encode_cat:
            new_dataset.speaker_encode = self.speaker_encode
            new_dataset.chapter_encode = self.chapter_encode
            new_dataset.utterance_encode = self.utterance_encode

            new_dataset.speaker = self.speaker_encode(new_dataset.speaker)
            new_dataset.chapter = self.chapter_encode(new_dataset.chapter)
            new_dataset.utterance = self.utterance_encode(new_dataset.utterance)

        return new_dataset

    def __exit__(self, exc_type, exc_value, traceback):
        if self.hpy_file is not None:
            self.hpy_file.close()

    def __getitem__(self, index):
        sound, speaker, chapter, utterance = self.sound[index], self.speaker[index], \
                                             self.chapter[index], self.utterance[index]
        sound = self.do_transform(sound)

        if self.one_hot_speaker or self.one_hot_all:
            speaker = self.do_one_hot(speaker, self.speaker_one_hot)
        if self.one_hot_chapter or self.one_hot_all:
            chapter = self.do_one_hot(chapter, self.chapter_one_hot)
        if self.one_hot_utterance or self.one_hot_all:
            utterance = self.do_one_hot(utterance, self.utterance_one_hot)

        return {"sound": sound, "speaker": speaker, "chapter": chapter, 'utterance': utterance}


if __name__ == "__main__":
    from mics.transforms import get_train_transform

    train_transforms = get_train_transform(length=2 ** 14)
    dataset = H5PyDataset("../librispeach/train-clean-100.hdf5",
                          transforms=train_transforms,
                          sr=16000,
                          one_hot_utterance=True,
                          in_memory=False)
    print("Dataset Len", len(dataset))
    print("item 0", dataset[0])

    dataset = dataset.instance_dataset("../librispeach/train-clean-100.hdf5", train_transforms, True)

    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 1}
    training_generator = data.DataLoader(dataset, **params)

    for batch in training_generator:
        print(batch['sound'].shape)
        print(batch['utterance'])
        break
