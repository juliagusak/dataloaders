from torch.utils.data import DataLoader

from nsynth import *


class NSynthTFRecordDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super(NSynthTFRecordDataLoader, self).__init__(dataset, **kwargs)

    def __iter__(self):
        audio = []
        pitch = []
        velocity = []
        instr_src = []
        instr_fml = []
        qualities = []

        for idx in range(len(self.dataset)):
            elem = self.dataset[idx]
            audio.append(elem[AUDIO])
            pitch.append(elem[PITCH])
            velocity.append(elem[VELOCITY])
            instr_src.append(elem[INSTR_SRC])
            instr_fml.append(elem[INSTR_FAMILY])
            qualities.append(elem[QUALITIES])

            if (idx + 1) % self.batch_size == 0:
                yield {AUDIO: np.vstack(audio), PITCH: np.hstack(pitch),
                       VELOCITY: np.hstack(velocity), INSTR_SRC: np.hstack(instr_src),
                       INSTR_FAMILY: np.hstack(instr_fml), QUALITIES: np.hstack(qualities)}

                audio.clear()
                pitch.clear()
                velocity.clear()
                instr_src.clear()
                instr_fml.clear()
                qualities.clear()

        return {AUDIO: np.vstack(audio), PITCH: np.hstack(pitch),
                VELOCITY: np.hstack(velocity), INSTR_SRC: np.hstack(instr_src),
                INSTR_FAMILY: np.hstack(instr_fml), QUALITIES: np.hstack(qualities)}


class NSynthTFRecordTestDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        kwargs['batch_size'] = 1
        super(NSynthTFRecordTestDataLoader, self).__init__(dataset, **kwargs)

    def __iter__(self):
        for idx in range(len(self.dataset)):
            elem = self.dataset[idx]
            result = {AUDIO: elem[AUDIO], PITCH: elem[PITCH],
                      VELOCITY: elem[VELOCITY], INSTR_SRC: elem[INSTR_SRC],
                      INSTR_FAMILY: elem[INSTR_FAMILY], QUALITIES: elem[QUALITIES]}
            yield result


if __name__ == "__main__":
    from misc.transforms import get_train_transform, get_test_transform

    params = {'batch_size': 64,
              'shuffle': False,
              'num_workers': 1}

    train_transforms = get_train_transform(length=2 ** 14)
    dataset = NSynthTFRecordDataset("../nsynth-test.tfrecord",
                                    one_hot_pitch=True,
                                    encode_cat=True,
                                    transforms=train_transforms,
                                    sr=16000,
                                    in_memory=False)
    test_generator = NSynthTFRecordDataLoader(dataset, **params)
    for batch in test_generator:
        print(batch['audio'].shape)
        print(batch)
        break

    print("--------------------------")
    params = {'batch_size': 64,
              'shuffle': False,
              'num_workers': 1}

    test_transforms = get_test_transform(length=2 ** 14)
    dataset = NSynthTFRecordDataset("../nsynth-test.tfrecord",
                                    one_hot_pitch=True,
                                    encode_cat=True,
                                    transforms=test_transforms,
                                    sr=16000,
                                    in_memory=False)
    test_generator = NSynthTFRecordTestDataLoader(dataset, **params)
    for batch in test_generator:
        print(batch['audio'].shape)
        print(batch)
        break
