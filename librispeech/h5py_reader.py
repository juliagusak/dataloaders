import h5py

import numpy as np

from torch.utils import data


from mics.utils import tensor_to_numpy
from librispeech.utils import SPEAKER, SOUND, CHAPTER, UTTERANCE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class LabelsToOneHot:
    def __init__(self, data):
        self.labels_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder()

        self.labels_encoder.fit(data.reshape(-1, ))
        self.one_hot_encoder.fit(self.labels_encoder.transform(data.reshape(-1, )).reshape((-1, 1)))

    def __call__(self, data):
        return self.one_hot_encoder.transform(self.labels_encoder.transform(data.reshape(-1, )).reshape((-1, 1)))


class LibriSpeechH5py(data.Dataset):
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
        self.in_memory = in_memory
        self.transforms = transforms
        self.sr = sr
        self.signal_length = signal_length
        self.precision = precision

        self.hpy_file = None
        with h5py.File(dataset_path, 'r') as f:
            if self.in_memory:
                self.sound = f[SOUND][:10]
            else:
                self.sound = f[SOUND]
            self.speaker, self.chapter, self.utterance = f[SPEAKER][:10], f[CHAPTER][:10], f[UTTERANCE][:10]

        self.n = self.speaker.shape[0]

        self.one_hot_all = one_hot_all
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
    from librispeech.transforms import get_train_transform

    train_transforms = get_train_transform(length=2 ** 14)
    dataset = LibriSpeechH5py("./librispeach/train-clean-100.hdf5",
                              transforms=train_transforms,
                              sr=16000,
                              one_hot_utterance=True)
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
