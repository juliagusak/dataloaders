import numpy as np
from torch.utils.data import DataLoader


class LibriSpeechTFRecordDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super(LibriSpeechTFRecordDataLoader, self).__init__(dataset, **kwargs)

    def __iter__(self):
        sound = []
        speaker = []
        label = []
        sr = []
        for idx in range(len(self.dataset)):

            elem = self.dataset[idx]
            sound.append(elem["sound"])
            speaker.append(elem["speaker"])
            label.append(elem["label"])
            sr.append(elem["sr"])

            if (idx + 1) % self.batch_size == 0:
                yield {"sound": np.vstack(sound), "speaker": np.hstack(speaker),
                       "label": np.hstack(label), "sr": np.hstack(sr)}

                sound.clear()
                speaker.clear()
                label.clear()
                sr.clear()

        batch = {"sound": np.vstack(sound), "speaker": np.hstack(speaker),
                 "label": np.hstack(label), "sr": np.hstack(sr)}
        yield batch


class LibriSpeechTFRecordTestDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        kwargs['batch_size'] = 1
        super(LibriSpeechTFRecordTestDataLoader, self).__init__(dataset, **kwargs)

    def __iter__(self):
        for idx in range(len(self.dataset)):
            elem = self.dataset[idx]
            result = {"sound": elem["sound"], "speaker": elem["speaker"],
                      "label": elem["label"], "sr": elem["sr"]}
            yield result


if __name__ == "__main__":
    from mics.transforms import get_train_transform, get_test_transform
    from librispeech.torch_readers.dataset_tfrecord import TFRecordDataset

    params = {'batch_size': 64,
              'shuffle': False,
              'num_workers': 1}

    dataset = TFRecordDataset("../librispeach/test-clean-100_wav16.tfrecord",
                              get_train_transform(16000), 16000, in_memory=False)
    test_generator = LibriSpeechTFRecordDataLoader(dataset, **params)
    for batch in test_generator:
        print(batch['sound'].shape)
        print(batch)
        break

    params = {'batch_size': 64,
              'shuffle': False,
              'num_workers': 1}

    dataset = TFRecordDataset("../librispeach/test-clean-100_wav16.tfrecord",
                              get_test_transform(16000), 16000, in_memory=False)
    test_generator = LibriSpeechTFRecordTestDataLoader(dataset, **params)
    for batch in test_generator:
        print(batch['sound'].shape)
        print(batch)
        break
