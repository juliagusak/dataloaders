import argparse
import subprocess

import h5py
import os

import numpy as np
import tensorflow as tf

NSYNTH_TRAIN = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.tfrecord"
NSYNTH_TEST = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.tfrecord"
NSYNTH_VAL = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.tfrecord"

TRAIN_FILE = NSYNTH_TRAIN.split("/")[-1]
TEST_FILE = NSYNTH_TEST.split("/")[-1]
VAL_FILE = NSYNTH_VAL.split("/")[-1]

TRAIN_EXAMPLES = 289205
VAL_EXAMPLES = 12678
TEST_EXAMPLES = 4096

AUDIO_LEN = 64000
QUALITIES_LEN = 10


def parse_args():
    parser = argparse.ArgumentParser(description='NSynth')

    # General settings
    parser.add_argument('--path', required=True, help="Where to store results")
    parser.add_argument('--train', action='store_true', help="Download the train dataset")
    parser.add_argument('--test', action='store_true', help="Download the train dataset")
    parser.add_argument('--val', action='store_true', help="Download the validate dataset")
    parser.add_argument('--force_download', action='store_true', help="Force downloading from website.")
    parser.add_argument('--force_h5py', action='store_true', help="Force creating h5py file.")
    parser.add_argument('--store_h5py', action='store_true', help="Forcing storing to h5py_utils")
    parser.add_argument('--batch_size', default=256, type=int, help="How many items read from tfrecord at once")

    return parser.parse_args()


def download_dataset(url, path, force_download):
    if not os.path.exists(path) or force_download:
        if force_download and os.path.exists(path):
            print("Force download. {} file will me replaced.".format(path))
            os.remove(path)

        print("Download *.tfrecord file to", path)
        subprocess.run("wget {} -P {}".format(url, path), shell=True, check=True)
    else:
        print("The dataset has been already downloaded to {}".format(path))


def _extract_features(example):
    features = {
        "note_str": tf.FixedLenFeature([], dtype=tf.string),
        "pitch": tf.FixedLenFeature([1], dtype=tf.int64),
        "velocity": tf.FixedLenFeature([1], dtype=tf.int64),
        "audio": tf.FixedLenFeature([64000], dtype=tf.float32),
        "qualities": tf.FixedLenFeature([10], dtype=tf.int64),
        "instrument_source": tf.FixedLenFeature([1], dtype=tf.int64),
        "instrument_family": tf.FixedLenFeature([1], dtype=tf.int64),
    }

    parsed_example = tf.parse_single_example(example, features)

    audio = tf.reshape(tf.cast(parsed_example["audio"], tf.float32), [1, 64000])
    pitch = tf.cast(parsed_example["pitch"], tf.int64)
    velocity = tf.cast(parsed_example["velocity"], tf.int64)
    instrument_source = tf.cast(parsed_example["instrument_source"], tf.int64)
    instrument_family = tf.cast(parsed_example["instrument_family"], tf.int64)
    qualities = tf.reshape(tf.cast(parsed_example["qualities"], tf.int64), [1, 10])
    return audio, pitch, velocity, instrument_source, instrument_family, qualities


def itarate_over_tfrecord(iter):
    with tf.Session() as sess:
        try:
            while True:
                yield sess.run(iter.get_next())
        except tf.errors.OutOfRangeError:
            pass


if __name__ == "__main__":
    opt = parse_args()

    process_files = []
    if opt.train:
        download_dataset(NSYNTH_TRAIN, opt.path, opt.force_download)
        process_files.append((os.path.join(opt.path, TRAIN_FILE), TRAIN_EXAMPLES))
    if opt.test:
        download_dataset(NSYNTH_TEST, opt.path, opt.force_download)
        process_files.append((os.path.join(opt.path, TEST_FILE), TEST_EXAMPLES))
    if opt.val:
        download_dataset(NSYNTH_VAL, opt.path, opt.force_download)
        process_files.append((os.path.join(opt.path, VAL_FILE), VAL_EXAMPLES))

    if opt.store_h5py:
        for file_name, num_examples in process_files:
            dataset_path = file_name[:-9] + ".hdf5"

            if opt.force_h5py and os.path.exists(dataset_path):
                print("h5py file {} will be removed".format(dataset_path))
                subprocess.run("rm -rf {}".format(dataset_path), shell=True, check=True)
            if not opt.force_h5py and os.path.exists(dataset_path):
                print("h5py file {} has been already created".format(dataset_path))
                continue

            dataset = tf.data.TFRecordDataset(file_name)
            dataset = dataset.map(_extract_features)
            dataset = dataset.batch(opt.batch_size)
            dataset = dataset.repeat(1)

            iter = dataset.make_one_shot_iterator()

            f = h5py.File(dataset_path, 'w')

            dt = h5py.special_dtype(vlen=np.float32)
            audio_ds = f.create_dataset("audio", (num_examples, AUDIO_LEN), dtype=np.float32)
            pitch_ds = f.create_dataset("pitch", (num_examples,), dtype=np.int)
            velocity_ds = f.create_dataset("velocity", (num_examples,), dtype=np.int)
            instr_src_ds = f.create_dataset("instrument_source", (num_examples,), dtype=np.int)
            instr_fml_ds = f.create_dataset("instrument_family", (num_examples,), dtype=np.int)
            qualities_ds = f.create_dataset("qualities", (num_examples, QUALITIES_LEN), dtype=np.int)

            idx = 0
            for audio, pitch, velocity, instrument_source, instrument_family, qualities in itarate_over_tfrecord(iter):
                curr_batch_size = audio.shape[0]
                start = idx
                end = idx + curr_batch_size
                idx = end

                audio_ds[start:end, :] = audio.reshape((audio.shape[0], -1))
                pitch_ds[start:end] = pitch.reshape((-1))
                velocity_ds[start:end] = velocity.reshape((-1))
                instr_src_ds[start:end] = instrument_source.reshape((-1))
                instr_fml_ds[start:end] = instrument_family.reshape((-1))
                qualities_ds[start:end, :] = qualities.reshape((-1, QUALITIES_LEN))
            f.close()
            print("Complete converting: {} to {}".format(file_name, dataset_path))
