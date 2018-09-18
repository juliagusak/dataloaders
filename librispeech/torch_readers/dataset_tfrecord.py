import numpy as np
import tensorflow as tf

from librispeech.torch_readers.constants import *
from misc.basic_dataset import BasicDataset
from misc.utils import LabelsToOneHot, LabelsEncoder, itarate_over_tfrecord, configure_tf_dataset


def librispeech_features(example):
    features = {
        'signal_raw': tf.FixedLenFeature([], tf.string),
        'sr': tf.FixedLenFeature([], tf.int64),
        'speaker': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    }

    parsed_example = tf.parse_single_example(example, features)

    sound = tf.decode_raw(parsed_example['signal_raw'], tf.float32)
    sr = tf.cast(parsed_example['sr'], tf.int32)
    speaker = tf.cast(parsed_example['speaker'], tf.int32)
    label = tf.cast(parsed_example['label'], tf.int32)

    return sound, sr, speaker, label


class TFRecordDataset(BasicDataset):
    def __init__(self,
                 dataset_path,
                 transforms,
                 sr,
                 signal_length=2 ** 16,
                 precision=np.float32,
                 one_hot_all=False,
                 one_hot_speaker=False,
                 one_hot_label=False,
                 encode_cat=False,
                 in_memory=True,
                 batch_size=1,
                 repeat=1,
                 buffer_size=10):
        super(TFRecordDataset, self).__init__(transforms, sr, signal_length, precision,
                                              one_hot_all, encode_cat, in_memory)

        self.sound = []
        self.speaker = []
        self.label = []

        self.dataset = configure_tf_dataset(librispeech_features, batch_size, buffer_size, dataset_path, repeat)

        self.sound = []
        self.sr = []
        self.speaker = []
        self.label = []
        self.sess = None
        self.iterator = None

        iter = self.dataset.make_one_shot_iterator()
        if self.in_memory:
            for sound, sr, speaker, label in itarate_over_tfrecord(iter):
                self.sound.append(sound)
                self.sr.append(sr)
                self.speaker.append(speaker)
                self.label.append(label)

            self.sound = np.vstack(self.sound)
            self.sr = np.hstack(self.sr)
            self.speaker = np.hstack(self.speaker)
            self.label = np.hstack(self.label)

            self.n = self.label.shape[0]
        else:
            self.sess = tf.Session()
            self.n = 0
            for sound, sr, speaker, label in itarate_over_tfrecord(iter):
                self.speaker.append(speaker[0])
                self.label.append(label[0])
                self.n += 1
            self.speaker = np.array(self.speaker)
            self.label = np.array(self.label)

        self.one_hot_speaker = one_hot_speaker
        self.one_hot_label = one_hot_label

        if self.encode_cat:
            self.speaker_encoder = LabelsEncoder(self.speaker)
            self.label_encoder = LabelsEncoder(self.label)

            self.speaker = self.speaker_encoder(self.speaker)
            self.label = self.label_encoder(self.label)
        else:
            self.speaker_encoder = None
            self.label_encoder = None

        if self.one_hot_speaker or self.one_hot_all:
            self.speaker_one_hot = LabelsToOneHot(self.speaker)
        else:
            self.speaker_one_hot = None

        if self.one_hot_label or self.one_hot_all:
            self.label_one_hot = LabelsToOneHot(self.label)
        else:
            self.label_one_hot = None

    def instance_dataset(self, dataset_path, transforms, in_memory):
        new_dataset = self.__class__(dataset_path,
                                     transforms,
                                     sr=self.sr,
                                     signal_length=self.signal_length,
                                     precision=self.precision,
                                     one_hot_all=False,
                                     one_hot_speaker=False,
                                     one_hot_label=False,
                                     encode_cat=False,
                                     in_memory=in_memory
                                     )

        new_dataset.one_hot_all = self.one_hot_all

        if self.one_hot_speaker or self.one_hot_all:
            new_dataset.one_hot_speaker = True
            new_dataset.speaker_one_hot = self.speaker_one_hot
        if self.one_hot_label or self.one_hot_all:
            new_dataset.one_hot_label = True
            new_dataset.label_one_hot = self.label_one_hot

        if self.encode_cat:
            new_dataset.speaker_encode = self.speaker_encoder
            new_dataset.label_encoder = self.label_encoder

        return new_dataset

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sess is not None:
            self.sess.close()

    def __getitem__(self, index):
        if index >= self.n:
            raise IndexError

        if self.in_memory:
            sound, sr, speaker, label = self.sound[index], self.sr[index], self.speaker[index], self.label[index]
        else:
            if self.iterator is None:
                self.iterator = self.dataset.make_one_shot_iterator()
            try:
                sound, sr, speaker, label = self.iterator.get_next()
                sound, sr, speaker, label = self.sess.run([sound, sr, speaker, label])
            except tf.errors.OutOfRangeError:
                self.iterator = self.dataset.make_one_shot_iterator()
                sound, sr, speaker, label = self.sess.run(self.iterator.get_next())

        sound = self.do_transform(sound)

        if self.encode_cat:
            speaker = self.speaker_encoder(speaker)
            label = self.label_encoder(label)

        if self.one_hot_all or self.one_hot_speaker:
            speaker = self.speaker_one_hot(speaker)
        if self.one_hot_all or self.one_hot_label:
            label = self.label_one_hot(label)

        return {SOUND: sound, SR: sr, SPEAKER: speaker, LABEL: label}


if __name__ == "__main__":
    from misc.transforms import get_train_transform

    train_transforms = get_train_transform(16000)
    dataset = TFRecordDataset("../librispeach/test-clean-100_wav16.tfrecord",
                              train_transforms, 16000, in_memory=False, encode_cat=True)
    print(dataset[3]['sound'].shape)
    print(len(dataset))
    i = 0
    for _ in dataset:
        i += 1
    print(i)

    print("------------------------------")
    dataset = dataset.instance_dataset("../librispeach/test-clean-100_wav16.tfrecord", train_transforms, False)

    print(dataset[3]['sound'].shape)
    print(len(dataset))

