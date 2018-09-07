from torch.utils.data import DataLoader


class LibriSpeechH5pyTestDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        kwargs['batch_size'] = 1
        super(LibriSpeechH5pyTestDataLoader, self).__init__(dataset, **kwargs)

    def __iter__(self):
        iterator = super(LibriSpeechH5pyTestDataLoader, self).__iter__()
        for batch in iterator:
            batch['sound'] = batch['sound'].view(batch['sound'].size()[1:])
            yield batch


if __name__ == "__main__":
    from mics.transforms import get_test_transform
    from librispeech.h5py_reader import LibriSpeechH5py

    test_transforms = get_test_transform(length=2 ** 14)
    test_dataset = LibriSpeechH5py("./librispeach/train-clean-100.hdf5",
                                   transforms=test_transforms,
                                   sr=16000,
                                   one_hot_utterance=True,
                                   in_memory=False)
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 1}
    test_generator = LibriSpeechH5pyTestDataLoader(test_dataset, **params)
    for batch in test_generator:
        print(batch['sound'].shape)
        print(batch)
        break
