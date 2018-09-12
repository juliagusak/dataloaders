import tensorflow as tf

from nsynth.constants import *


def nsynth_extract_features(example):
    features = {
        NOTE_STR: tf.FixedLenFeature([], dtype=tf.string),
        PITCH: tf.FixedLenFeature([1], dtype=tf.int64),
        VELOCITY: tf.FixedLenFeature([1], dtype=tf.int64),
        AUDIO: tf.FixedLenFeature([64000], dtype=tf.float32),
        QUALITIES: tf.FixedLenFeature([10], dtype=tf.int64),
        INSTR_SRC: tf.FixedLenFeature([1], dtype=tf.int64),
        INSTR_FAMILY: tf.FixedLenFeature([1], dtype=tf.int64),
    }

    parsed_example = tf.parse_single_example(example, features)

    audio = tf.reshape(tf.cast(parsed_example[AUDIO], tf.float32), [1, 64000])
    pitch = tf.cast(parsed_example[PITCH], tf.int64)
    velocity = tf.cast(parsed_example[VELOCITY], tf.int64)
    instrument_source = tf.cast(parsed_example[INSTR_SRC], tf.int64)
    instrument_family = tf.cast(parsed_example[INSTR_FAMILY], tf.int64)
    qualities = tf.reshape(tf.cast(parsed_example[QUALITIES], tf.int64), [1, 10])
    return audio, pitch, velocity, instrument_source, instrument_family, qualities