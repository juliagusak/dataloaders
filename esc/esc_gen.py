"""
 Dataset preparation code for ESC-50 and ESC-10 [Piczak, 2015].
 Usage: python esc_gen.py [path]
 FFmpeg should be installed.

"""

import sys
import os
import subprocess

import glob
import numpy as np
import wavio


def main():
    esc50_path = os.path.join(sys.argv[1], 'esc50')
    esc10_path = os.path.join(sys.argv[1], 'esc10')
    os.mkdir(esc50_path)
    os.mkdir(esc10_path)
    fs_list = [16000, 44100]  # EnvNet and EnvNet-v2, respectively

    # Download ESC-50
    subprocess.call('wget -P {} https://github.com/karoldvl/ESC-50/archive/master.zip'.format(
        esc50_path), shell=True)
    subprocess.call('unzip -d {} {}'.format(
        esc50_path, os.path.join(esc50_path, 'master.zip')), shell=True)
    os.remove(os.path.join(esc50_path, 'master.zip'))

    # Convert sampling rate
    for fs in fs_list:
        if fs == 44100:
            continue
        else:
            convert_fs(os.path.join(esc50_path, 'ESC-50-master', 'audio'),
                       os.path.join(esc50_path, 'wav{}'.format(fs // 1000)),
                       fs)

    # Create npz files
    for fs in fs_list:
        if fs == 44100:
            src_path = os.path.join(esc50_path, 'ESC-50-master', 'audio')
        else:
            src_path = os.path.join(esc50_path, 'wav{}'.format(fs // 1000))

        create_dataset(src_path,
                       os.path.join(esc50_path, 'wav{}.npz'.format(fs // 1000)),
                       os.path.join(esc10_path, 'wav{}.npz'.format(fs // 1000)))


def convert_fs(src_path, dst_path, fs):
    print('* {} -> {}'.format(src_path, dst_path))
    os.mkdir(dst_path)
    for src_file in sorted(glob.glob(os.path.join(src_path, '*.wav'))):
        dst_file = src_file.replace(src_path, dst_path)
        subprocess.call('ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(
            src_file, fs, dst_file), shell=True)


def create_dataset(src_path, esc50_dst_path, esc10_dst_path):
    print('* {} -> {}'.format(src_path, esc50_dst_path))
    print('* {} -> {}'.format(src_path, esc10_dst_path))
    esc10_classes = [0, 10, 11, 20, 38, 21, 40, 41, 1, 12]  # ESC-10 is a subset of ESC-50
    esc50_dataset = {}
    esc10_dataset = {}

    for fold in range(1, 6):
        esc50_dataset['fold{}'.format(fold)] = {}
        esc50_sounds = []
        esc50_labels = []
        esc10_dataset['fold{}'.format(fold)] = {}
        esc10_sounds = []
        esc10_labels = []

        for wav_file in sorted(glob.glob(os.path.join(src_path, '{}-*.wav'.format(fold)))):
            sound = wavio.read(wav_file).data.T[0]
            start = sound.nonzero()[0].min()
            end = sound.nonzero()[0].max()
            sound = sound[start: end + 1]  # Remove silent sections
            label = int(os.path.splitext(wav_file)[0].split('-')[-1])
            esc50_sounds.append(sound)
            esc50_labels.append(label)
            if label in esc10_classes:
                esc10_sounds.append(sound)
                esc10_labels.append(esc10_classes.index(label))

        esc50_dataset['fold{}'.format(fold)]['sounds'] = esc50_sounds
        esc50_dataset['fold{}'.format(fold)]['labels'] = esc50_labels
        esc10_dataset['fold{}'.format(fold)]['sounds'] = esc10_sounds
        esc10_dataset['fold{}'.format(fold)]['labels'] = esc10_labels

    np.savez(esc50_dst_path, **esc50_dataset)
    np.savez(esc10_dst_path, **esc10_dataset)


if __name__ == '__main__':
    main()
