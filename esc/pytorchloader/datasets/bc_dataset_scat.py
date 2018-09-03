import os
import numpy as np
# import torchvision.transforms as transforms

from torch.utils import data

from .transforms import get_train_transform, get_test_transform

from .utils import IMG, LABEL, numpy_one_hot, mix, tensor_to_numpy



class BCDatasets(data.Dataset):
    def __init__(self, data_path, dataset_name,
                 sr, exclude, 
                 is_train = True,
                 signal_length = 2**16,
                 mix=False, precision = np.float32):
        
        self.signal_length = signal_length
        
        if is_train:
            self.transform = get_train_transform(length = signal_length)
        else:
            self.transform = get_test_transform(length = signal_length)
                                                
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
        
#         print(np.max(list(map(len, self.X))))
#         print(np.mean(list(map(len, self.X))))
#         print(np.min(list(map(len, self.X))))
    
    
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __do_transform(self, x, y):
        # __do_transform operates with 1d signals
        # return numpy array of shape (1, 1, signal_length)
        y = numpy_one_hot(y, num_classes=self.n_classes)

        x = x.astype(self.precision)
        sample = {IMG: x, LABEL: y}
        if self.transform:
            sample[IMG] = sample[IMG].reshape((1, -1, 1))
            sample = self.transform(sample)
            sample[IMG] = tensor_to_numpy(sample[IMG])
#             print(sample[IMG].shape)
                       
        return sample
    

    def __mix_samples(self, sample1, sample2):
        r = np.random.uniform()

        sound1 = sample1[IMG].reshape((-1))
        sound2 = sample2[IMG].reshape((-1))

        sound = mix(sound1, sound2, r, self.sr)
        label = r * sample1[LABEL] + (1.0 - r) * sample2[LABEL]

        return {IMG: sound, LABEL: label}

    def __getitem__(self, index):
        if self.mix:
            idx1, idx2 = np.random.choice(len(self), 2, replace=False)

            sound1, label1 = self.X[idx1], self.y[idx1]
            sound2, label2 = self.X[idx2], self.y[idx2]

            sample1 = self.__do_transform(sound1, label1)
            sample2 = self.__do_transform(sound2, label2)

            sample = self.__mix_samples(sample1, sample2)
        else:
            sample = self.__do_transform(self.X[index], self.y[index])
            
        # can reshape to the numpy array of shape (signal_length, )     
        # sample[IMG] = sample[IMG].reshape((1, 1, -1))

        return sample

# if __name__== "__main__":
#     data_path = "/home/julia/DeepVoice_data/ESC"
#     dataset_name="esc10"
#     sr=16000
#     exclude=[5]
#     dataset = BCDatasets(data_path, dataset_name, sr, exclude, scattering_time_transform = False)
#     print(dataset[0])
