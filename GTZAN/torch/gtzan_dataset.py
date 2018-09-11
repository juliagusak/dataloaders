import numpy as np
from torch.utils import data

from mics.data_loader import ValidationDataLoader
from mics.transforms import get_train_transform, get_test_transform
from mics.utils import LabelsToOneHot, tensor_to_numpy


class GTANZDataset(data.Dataset):
    def __init__(self, dataset_path, transforms=None, one_hot_labels=False):
        data = np.load(dataset_path)
        self.transforms = transforms
        self.X = data["X"]
        self.y = data["y"]
        self.label_name = data["label_name"]

        self.n = self.X.shape[0]

        self.one_hot_labels = one_hot_labels
        if one_hot_labels:
            self.one_hot_encoder = LabelsToOneHot(self.y)
        else:
            self.one_hot_encoder = None

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        X, y, label_name = self.X[index], self.y[index], self.label_name[index]

        if self.transforms:
            X = tensor_to_numpy(self.transforms(X.reshape((1, -1, 1))))

        if self.one_hot_labels:
            y = self.one_hot_encoder(y).toarray()[0, :]

        return {"sound": X, "class": y, "class_label": label_name}


if __name__=="__main__":
    dataset = GTANZDataset("../genres16_test.npz",
                           transforms=get_train_transform(length=2 ** 14),
                           one_hot_labels=True)
    print(len(dataset))
    print(dataset[5])

    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 1}
    dataset = GTANZDataset("../genres16_test.npz",
                           transforms=get_test_transform(length=2 ** 14),
                           one_hot_labels=True)
    test_generator = ValidationDataLoader(dataset, **params)
    for batch in test_generator:
        print(batch['sound'].shape)
        print(batch)
        break
