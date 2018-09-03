import tensorflow as tf
import numpy as np
import esc_utils as U

def _bytes_features(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _int64_features(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))


def write_tfrecords(tfrecord_path, sounds, labels, fs = 16000):    
    with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
    
        for sound, label in zip(sounds, labels):
            sound_raw = sound.tostring()

            example = tf.train.Example(features = tf.train.Features(
                feature = {
                    'signal_raw': _bytes_features(sound_raw),
                    'sr': _int64_features(fs),
                    'speaker': _int64_features(label),
                    'label': _int64_features(label)
                }))
            writer.write(example.SerializeToString())
        
        

def create_tfrecords(npz_path, tfrecord_pathes,
                     split = 4, fs = 16000,
                     augment = False, strong = False):
# tfrecord_pathes = pathes for train, val tfrecords    
    with np.load(npz_path) as dataset:

        train_sounds, train_labels = [], []
        val_sounds, val_labels = [], []

        for i, fold in enumerate(dataset.files):
            sounds = dataset[fold].item()['sounds']
            labels = dataset[fold].item()['labels']
            
            
            # we'll add to dataset only samples with length >= 40000
            idxs = list(filter(lambda i: len(sounds[i]) >= 40000,
                               range(len(sounds))))
            sounds = list(np.array(sounds)[idxs])
            labels = list(np.array(labels)[idxs])
            
            print('Preprocessing sounds...')
            sounds = [U.preprocess_sound(sound) for sound in sounds]
#             print(len(sounds), len(labels))            
                
            print('Augmenting data...')
            if augment:
                augmented_sounds, augmented_labels = [], []
                for sound, label in zip(sounds, labels):
                    augmented_sounds.extend([U.augment_sound(sound, strong = strong) for _ in range(9)])
                    augmented_labels.extend([label]*9)
                    
                sounds.extend(augmented_sounds)
                labels.extend(augmented_labels)                    
            
            if i  == split:
                val_sounds.extend(sounds)
                val_labels.extend(labels)
            else:
                train_sounds.extend(sounds)
                train_labels.extend(labels)
                
        print(len(train_sounds), len(train_labels))
        print(len(val_sounds), len(val_labels))

        print('Writing tfrecords...')
        train_tfrecord_path, val_tfrecord_path = tfrecord_pathes

        write_tfrecords(train_tfrecord_path, train_sounds, train_labels, fs = fs)
        write_tfrecords(val_tfrecord_path, val_sounds, val_labels, fs = fs)

                
            
if __name__ == "__main__":
    FS = 16000
    SPLIT = 4
    AUGMENT = True
    STRONG = True
    
    esc_path = '/workspace/data/ESC/esc10/'
    
    npz_path = '{}wav{}.npz'.format(esc_path, FS//1000)
    tfrecord_pathes = ['{}wav{}_train.tfrecord'.format(esc_path, FS//1000),
                       '{}wav{}_val.tfrecord'.format(esc_path, FS//1000)]
    
    
    create_tfrecords(npz_path, tfrecord_pathes,
                     split = SPLIT, fs = FS,
                     augment = AUGMENT, strong = STRONG)