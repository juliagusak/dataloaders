import os
import numpy as np

from torch.utils import data

from misc.transforms import get_train_transform, get_test_transform
from misc.utils import FEATURES, LABEL, numpy_one_hot, mix, tensor_to_numpy


class ESCDatasets(data.Dataset):
    def __init__(self, data_path, dataset_name,
                 sr, exclude,
                 is_train=True,
                 signal_length=2 ** 16,
                 mix=False, precision=np.float32):

        self.signal_length = signal_length

        if is_train:
            self.transform = get_train_transform(length=signal_length)
        else:
            self.transform = get_test_transform(length=signal_length)

        self.sr = sr
        self.mix = mix
        self.precision = precision
        data_set = np.load(os.path.join(data_path, dataset_name, 'wav{}.npz'.format(sr // 1000)))

        self.X = []
        self.y = []
        for fold_name in data_set.keys():
            if int(fold_name[4:]) in exclude:
                continue

            sounds = data_set[fold_name].item()['sounds']
            labels = data_set[fold_name].item()['labels']

            self.X.extend(sounds)
            self.y.extend(labels)

        self.n_classes = len(set(self.y))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __do_transform(self, signal):
        signal = signal.astype(self.precision)
        if self.transform:
            signal = tensor_to_numpy(self.transform(signal.reshape((1, -1, 1))))

        return signal

    def __mix_samples(self, sample1, sample2):
        r = np.random.uniform()

        sound1 = sample1[FEATURES].reshape((-1))
        sound2 = sample2[FEATURES].reshape((-1))

        sound = mix(sound1, sound2, r, self.sr)
        label = r * sample1[LABEL] + (1.0 - r) * sample2[LABEL]

        return {FEATURES: sound, LABEL: label}

    def __getitem__(self, index):
        if self.mix:
            idx1, idx2 = np.random.choice(len(self), 2, replace=False)

            sample1 = {FEATURES: self.__do_transform(self.X[idx1]),
                       LABEL: numpy_one_hot(self.y[idx1], num_classes=self.n_classes)}
            sample2 = {FEATURES: self.__do_transform(self.X[idx2]),
                       LABEL: numpy_one_hot(self.y[idx2], num_classes=self.n_classes)}

            sample = self.__mix_samples(sample1, sample2)

        else:
            sample = {FEATURES: self.__do_transform(self.X[index]),
                      LABEL: self.y[index]}

        return sample


# if __name__ == "__main__":
#     data_path = "/home/julia/DeepVoice_data/ESC"
#     dataset_name = "esc10"
#     sr = 16000
#     exclude = [5]
#     dataset = BCDatasets(data_path, dataset_name, sr, exclude, scattering_time_transform=False)
#     print(dataset[0])
