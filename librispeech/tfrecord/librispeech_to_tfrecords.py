import tensorflow as tf
from glob import glob
import numpy as np
import librosa
import scipy

# create .tfrecords file with signals and annonation info 

def _bytes_features(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _int64_features(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def convert_speakers_to_labels(labels, convertion_dict = None):
    return np.array([convertion_dict[l] for l in labels], dtype = np.int32)

def write_tfrecords(wav_path, tfrecord_path, signal_length = 40000, sr = 16000):
    wav_files = glob('{}/**/*.wav'.format(wav_path), recursive=True)

    speakers = [int(file.split('/')[-1].split('-')[0]) for file in wav_files]
    speaker_to_label = {v:k for k,v in enumerate(set(speakers))}
    
    labels = convert_speakers_to_labels(speakers, convertion_dict = speaker_to_label)

    tfrecords_filename = tfrecord_path

    with tf.python_io.TFRecordWriter(tfrecords_filename) as writer:

        original_signals = []

        for wav_file, speaker, label in zip(wav_files, speakers, labels):
            print(wav_file)
            #sr, wav = scipy.io.wavfile.read(wav_file)
            wav, sr = librosa.core.load(wav_file, sr = sr, dtype = np.float32)

            if len(wav)<signal_length:
                continue
            else:
                wav = wav[:signal_length]

            annotation = (sr, speaker, label)
            original_signals.append((wav, annotation)) 

            # encode to bytes
            wav_raw = wav.tostring()

            example = tf.train.Example(features = tf.train.Features(
                feature = {
                    'signal_raw': _bytes_features(wav_raw),
                    'sr': _int64_features(sr),
                    'speaker': _int64_features(speaker),
                    'label': _int64_features(label)
                }))
            writer.write(example.SerializeToString())


if __name__=="__main__":
    for folder in ['train']:
        wav_path = '/workspace/data/LibriSpeech_to_classify/{}'.format(folder)
        tfrecord_path = '{}/wavs.tfrecord'.format(wav_path)

        write_tfrecords(wav_path, tfrecord_path)
