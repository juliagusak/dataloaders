import torch

import torchvision

from torchvision.transforms import ToPILImage, Pad, RandomCrop, ToTensor, TenCrop, Lambda

# MAX_INT = 32768.0
MAX_INT = 1.0


class Centring(object):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return img / self.factor


def get_train_transform(length=None):
    transforms = [ToPILImage(),
                  Pad((length // 2, 0)),
                  RandomCrop((1, length)),
                  ToTensor(),
                  Centring(MAX_INT)]
    return torchvision.transforms.Compose(transforms)


def get_test_transform(length=None):
    transforms = [ToPILImage(),
                  Pad((length // 2, 0)),
                  TenCrop((1, length)),
                  Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
                  Centring(MAX_INT)]
    return torchvision.transforms.Compose(transforms)
