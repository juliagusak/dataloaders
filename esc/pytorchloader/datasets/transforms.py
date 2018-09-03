import torch
import torchvision.transforms

from torchvision.transforms import TenCrop, Pad, RandomCrop, ToTensor, ToPILImage, Lambda
from .utils import IMG, LABEL, MAX_INT


class Centring(object):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return img / self.factor


class ConvertToTuple(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        img, label = sample[IMG], sample[LABEL]
        return {IMG: self.transform(img), LABEL: label}


def get_train_transform(length = None):
    transforms = [ToPILImage(),
                  Pad((length // 2, 0)),
                  RandomCrop((1, length)),
                  ToTensor(),
                  Centring(MAX_INT)]
    return torchvision.transforms.Compose([ConvertToTuple(default_transforms) for default_transforms in transforms])


def get_test_transform(length=None):
    transforms = [ToPILImage(),
                  Pad((length // 2, 0)),
                  TenCrop((1, length)),
#                   RandomCrop((1, length)),
#                   ToTensor(),
                  Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
                  Centring(MAX_INT)]
    return torchvision.transforms.Compose([ConvertToTuple(default_transforms) for default_transforms in transforms])