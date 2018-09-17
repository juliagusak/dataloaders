import numpy as np

from mics import configure_tf_dataset, itarate_over_tfrecord
from nsynth.torch_readers.basic_dataset import NSynthBasicDataset
from nsynth.utils import *


class NSynthTFRecordDataset(NSynthBasicDataset):
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
                 in_memory=True,
                 batch_size=1,
                 repeat=1,
                 buffer_size=10):
        # self.sess = tf.Session()
        self.sess = None
        self.iterator = None

        self.dataset = configure_tf_dataset(nsynth_extract_features, batch_size, buffer_size, dataset_path, repeat)

        super(NSynthTFRecordDataset, self).__init__(dataset_path,
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
        iter = self.dataset.make_one_shot_iterator()
        self.n = 0
        for audio, pitch, velocity, instrument_source, instrument_family, qualities in itarate_over_tfrecord(iter):
            if not self.in_memory:
                self.audio.append(audio)
            self.n += 1
            self.pitch.append(pitch)
            self.velocity.append(velocity)
            self.instr_src.append(instrument_source)
            self.instr_fml.append(instrument_family)
            self.qualities.append(qualities)

        self.audio = np.array(self.audio)
        self.pitch = np.array(self.pitch)
        self.velocity = np.array(self.velocity)
        self.instr_src = np.array(self.instr_src)
        self.instr_fml = np.array(self.instr_fml)
        self.qualities = np.array(self.qualities)

        if not self.in_memory:
            self.sess = tf.Session()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sess is not None:
            self.sess.close()

    def read_elem(self, index):
        if self.in_memory:
            audio, pitch, velocity = self.audio[index], self.pitch[index], self.velocity[index]
            instrument_source, instrument_family = self.instr_src[index], self.instr_fml[index]
            qualities = self.qualities[index]
        else:
            if self.iterator is None:
                self.iterator = self.dataset.make_one_shot_iterator()
            try:
                audio, pitch, velocity, instrument_source, instrument_family, qualities = self.sess.run(
                    self.iterator.get_next())
            except tf.errors.OutOfRangeError:
                self.iterator = self.dataset.make_one_shot_iterator()
                audio, pitch, velocity, instrument_source, instrument_family, qualities = self.sess.run(
                    self.iterator.get_next())

        return audio, pitch, velocity, instrument_source, instrument_family, qualities


if __name__ == "__main__":
    from mics.transforms import get_train_transform

    train_transforms = get_train_transform(length=2 ** 14)
    dataset = NSynthTFRecordDataset("../nsynth-test.tfrecord",
                                    one_hot_pitch=True,
                                    encode_cat=True,
                                    transforms=train_transforms,
                                    sr=16000,
                                    in_memory=False)
    print("Dataset Len", len(dataset))
    print("item 0", dataset[0])

    dataset = dataset.instance_dataset("../nsynth-test.tfrecord", train_transforms, False)

    print("Dataset Len", len(dataset))
    print("item 0", dataset[0])
