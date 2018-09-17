import numpy as np
import tensorflow as tf
import torch

from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

FEATURES = 'features'
LABEL = 'label'

MAX_INT = 32768.0

Image.MAX_IMAGE_PIXELS = None


class LabelsToOneHot:
    def __init__(self, data):
        self.labels_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder()

        self.labels_encoder.fit(data.reshape(-1, ))
        self.one_hot_encoder.fit(self.labels_encoder.transform(data.reshape(-1, )).reshape((-1, 1)))

    def __call__(self, data):
        return self.one_hot_encoder.transform(self.labels_encoder.transform(data.reshape(-1, )).reshape((-1, 1))).toarray()


class LabelsEncoder:
    def __init__(self, data):
        self.labels_encoder = LabelEncoder()
        self.labels_encoder.fit(data.reshape(-1, ))

    def __call__(self, data):
        return self.labels_encoder.transform(data.reshape(-1, ))


def configure_tf_dataset(features_extractor, batch_size, buffer_size, dataset_path, repeat):
        dataset = tf.data.TFRecordDataset(dataset_path)
        dataset = dataset.map(features_extractor)
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        return dataset.repeat(repeat)


def itarate_over_tfrecord(iter):
    iter = iter.get_next()
    with tf.Session() as sess:
        try:
            while True:
                yield sess.run(iter)
        except tf.errors.OutOfRangeError:
            pass


def tensor_to_numpy(tensor):
    if torch.cuda.is_available():
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()


def numpy_one_hot(label, num_classes=2):
    label = np.eye(num_classes)[label]
    return label


# For BC learning
def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
                           - np.log10(freq_sq + 12194 ** 2)
                           - np.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)

    return weight


def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
    if fs == 16000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2

    gain = []
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == 'RMSE':
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == 'A_weighting':
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)

    return gain_db


def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))

    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))

    return sound

