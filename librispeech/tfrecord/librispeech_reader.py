"""Module to load the Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import numpy as np
import tensorflow as tf

class LibriSpeechDataset(object):
    '''Dataset object to help manage the TFRecord loading.'''
    
    def __init__(self, tfrecord_path, is_training = True):
        self.is_training = is_training
        self.record_path = tfrecord_path
        
    def get_example(self, batch_size):
        """Get a single example from the tfrecord file.
        Args:
          batch_size: Int, minibatch size.
        Returns:
          tf.Example protobuf parsed from tfrecord.
        """
        reader = tf.TFRecordReader()
        num_epochs = None if self.is_training else 1
        capacity = batch_size

        path_queue = tf.train.input_producer(
            [self.record_path],
            num_epochs = num_epochs,
            shuffle = self.is_training,
            capacity = capacity)

        _, serialized_example = reader.read(path_queue)
        features = {
           'signal_raw': tf.FixedLenFeature([], tf.string),
            'sr': tf.FixedLenFeature([], tf.int64),
            'speaker': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
        example = tf.parse_single_example(serialized_example, features)
        return example
    
    def get_wavenet_batch(self, batch_size, length = 40000):
        '''Get the Tensor expression from the reader.
        Args:
          batch_size: The integer batch size.
          length: Number of timesteps of a cropped sample to produce.
        Returns:
          A dict of key:tensor pairs. This includes "speaker", "label", "wav", and "sr".
        '''
        example = self.get_example(batch_size)
        
        signal = tf.decode_raw(example['signal_raw'], tf.float32)
        sr = tf.cast(example['sr'], tf.int32)
        speaker = tf.cast(example['speaker'], tf.int32)
        label = tf.cast(example['label'], tf.int32)  
        
        annotation = (sr, speaker, label)
        
        if self.is_training:
          # random crop
            crop = tf.random_crop(signal, [length])
            crop = tf.reshape(crop, [1, length])

        else:
            # fixed center crop
            offset = (40000 - length) // 2  # 24320
            crop = tf.slice(signal, [offset], [length])
            crop = tf.reshape(crop, [1, length])
            
        crops, annotations = tf.train.shuffle_batch(
              [crop, annotation],
              batch_size,
              num_threads=4,
              capacity=500 * batch_size,
              min_after_dequeue=200 * batch_size)    
               
        crops = tf.reshape(tf.cast(crops, tf.float32), [batch_size, length])
        
        return {"wav": crops, "sr": annotations[:,0], "speaker": annotations[:,1], "label": annotations[:,2]}
