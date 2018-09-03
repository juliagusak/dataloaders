import numpy as np
import random

INPUT_LENGTH = 40000
FACTOR = 32768.0


# Default data augmentation
def padding(pad):
    def f(sound):
        return np.pad(sound, pad, 'constant')

    return f


def random_crop(size):
    def f(sound):
        org_size = len(sound)
        start = random.randint(0, org_size - size)
        return sound[start: start + size]

    return f


def normalize(factor):
    def f(sound):
        return sound / factor

    return f

# For strong data augmentation
# Scale audio signal (compress/decompress in time domain)
# For augmentation use scale from [0.8, 1.25]
def random_scale(max_scale, interpolate='Linear'):
    def f(sound):
        scale = np.power(max_scale, random.uniform(-1, 1))
        output_size = int(len(sound) * scale)
        ref = np.arange(output_size) / scale
        if interpolate == 'Linear':
            ref1 = ref.astype(np.int32)
            ref2 = np.minimum(ref1 + 1, len(sound) - 1)
            r = ref - ref1
            scaled_sound = sound[ref1] * (1 - r) + sound[ref2] * r
        elif interpolate == 'Nearest':
            scaled_sound = sound[ref.astype(np.int32)]
        else:
            raise Exception('Invalid interpolation mode {}'.format(interpolate))

        return scaled_sound

    return f


# Make audio louder / quieter
# For augmentation use db=6
def random_gain(db):
    def f(sound):
        return sound * np.power(10, random.uniform(-db, db) / 20.0)

    return f


def preprocess_sound(sound):
    sound = padding(INPUT_LENGTH//2)(sound)
    sound = random_crop(INPUT_LENGTH)(sound)
    sound = normalize(FACTOR)(sound)
    
    return sound.astype(np.float32)


def augment_sound(sound, strong = True):
    sound = random_scale(1.25)(sound)
    sound = preprocess_sound(sound)
    
    if strong:
        sound = random_gain(6)(sound)
        
    return sound.astype(np.float32)
