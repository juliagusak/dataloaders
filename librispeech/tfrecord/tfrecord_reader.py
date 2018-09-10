import numpy as np
import tensorflow as tf

from librispeech.basic_reader import LibriSpeechBasic
from mics.transforms import get_train_transform
from mics.utils import LabelsToOneHot


def _extract_features(example):
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


class LibriSpeechTFRecord(LibriSpeechBasic):
    def __init__(self,
                 dataset_path,
                 transforms,
                 sr,
                 signal_length=2 ** 16,
                 precision=np.float32,
                 one_hot_all=False,
                 one_hot_speaker=False,
                 one_hot_label=False,
                 in_memory=True,
                 batch_size=1,
                 repeat=1,
                 buffer_size=50):
        super(LibriSpeechTFRecord, self).__init__(transforms, sr, signal_length, precision, one_hot_all, in_memory)

        self.sound = []
        self.speaker = []
        self.label = []

        self.dataset = self.configure_tf_dataset(batch_size, buffer_size, dataset_path, repeat)

        self.sound = []
        self.sr = []
        self.speaker = []
        self.label = []
        self.sess = None
        self.iterator = None
        if self.in_memory:
            iter = self.dataset.make_one_shot_iterator()
            for sound, sr, speaker, label in self.itarate_over_tfrecord(iter):
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
            iter = self.dataset.make_one_shot_iterator()
            for sound, sr, speaker, label in self.itarate_over_tfrecord(iter):
                self.speaker.append(speaker)
                self.label.append(label)
                self.n += 1
            self.speaker = np.array(self.speaker)
            self.label = np.array(self.label)

        self.one_hot_speaker = one_hot_speaker
        self.one_hot_label = one_hot_label

        if self.one_hot_speaker or self.one_hot_all:
            self.speaker_encoder = LabelsToOneHot(self.speaker)
        else:
            self.speaker_encoder = None

        if self.one_hot_label or self.one_hot_all:
            self.label_encoder = LabelsToOneHot(self.label)
        else:
            self.label_encoder = None

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sess is not None:
            self.sess.close()

    def itarate_over_tfrecord(self, iter):
        with tf.Session() as sess:
            try:
                i = 0
                while True and i < 10:
                    i += 1
                    yield sess.run(iter.get_next())
            except tf.errors.OutOfRangeError:
                pass

    def configure_tf_dataset(self, batch_size, buffer_size, dataset_path, repeat):
        dataset = tf.data.TFRecordDataset(dataset_path)
        dataset = dataset.map(_extract_features)
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        return dataset.repeat(repeat)

    def __getitem__(self, index):
        if self.in_memory:
            result = {"sound": self.sound[index], "sr": self.sr[index],
                      "speaker": self.speaker[index], "label": self.label[index]}
        else:
            if self.iterator is None:
                self.iterator = self.dataset.make_one_shot_iterator()
            try:
                sound, sr, speaker, label = self.sess.run(self.iterator.get_next())
                result = {"sound": sound, "sr": sr, "speaker": speaker, "label": label}
            except tf.errors.OutOfRangeError:
                self.iterator = self.dataset.make_one_shot_iterator()
                sound, sr, speaker, label = self.sess.run(self.iterator.get_next())
                result = {"sound": sound, "sr": sr, "speaker": speaker, "label": label}

        result["sound"] = self.do_transform(result["sound"])
        if self.one_hot_all or self.one_hot_speaker:
            result["speaker"] = self.do_one_hot(result["speaker"], self.speaker_encoder)
        if self.one_hot_all or self.one_hot_label:
            result["label"] = self.do_one_hot(result["label"], self.label_encoder)

        return result


if __name__ == "__main__":
    dataset = LibriSpeechTFRecord("../librispeach/train-clean-100_wav16.tfrecord",
                                  get_train_transform(16000), 16000, in_memory=False)
    print(dataset[3]['sound'].shape)
    print(len(dataset))
