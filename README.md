# dataloaders
Pytorch and TFRecords data loaders for  several audio datasets

**Datasets**
1. [ESC](https://github.com/karoldvl/ESC-50) - dataset of environmental sounds
  - [x] [ESC Downloader](https://github.com/juliagusak/dataloaders/blob/master/esc/esc_gen.py)
  - [x] [Pytorch DataSet](https://github.com/juliagusak/dataloaders/blob/master/esc/pytorchloader/datasets/esc_dataset_scat.py)
  - [x] [TFRecords Loader](https://github.com/juliagusak/dataloaders/blob/master/esc/tfrecord/esc_reader.py)
  
2. [LibriSpeech](http://www.openslr.org/12/) - corpus of read English speech
  - [x] [LibriSpeech downloader for PyTorch](https://github.com/juliagusak/dataloaders/blob/master/librispeech/h5py_torch/librispeech_gen.py) 
  - [x] [PyTorch DataSet](https://github.com/juliagusak/dataloaders/blob/master/librispeech/h5py_torch/h5py_dataset.py)
  - [x] [PyTorch DataSet for TFRecord](https://github.com/juliagusak/dataloaders/blob/master/librispeech/tfrecord/librispeech_reader.py)
  - [x] [PyTorch DataLoaders for TFRecord](https://github.com/juliagusak/dataloaders/blob/master/librispeech/tfrecord/tfrecord_dataloader.py)
  - [x] [TFRecords Loader](https://github.com/juliagusak/dataloaders/blob/master/librispeech/tfrecord/tfrecord_reader.py)
  - [x] [TFRecords Generator](https://github.com/juliagusak/dataloaders/blob/master/librispeech/tfrecord/librispeech_to_tfrecords.py)
3. [NSynth](https://magenta.tensorflow.org/datasets/nsynth) - dataset of annotated musical notes
  - [x] [NSynth downloader and generator of *.h5py and *.tfrecord formats](https://github.com/juliagusak/dataloaders/blob/master/nsynth/nsynth_gen.py)
  - [x] [TFRecord reader](https://github.com/juliagusak/dataloaders/blob/master/nsynth/tfrecord/nsynth_reader.py)
  - [x] [PyTorch Dataset](https://github.com/juliagusak/dataloaders/blob/master/nsynth/torch_readers/dataset_h5py.py)
  - [x] [PyTorch Dataset for TFrecord](https://github.com/juliagusak/dataloaders/blob/master/nsynth/torch_readers/dataset_tfrecord.py)
  - [x] [PyTorch DataLoaders for TFRecord](https://github.com/juliagusak/dataloaders/blob/master/nsynth/torch_readers/dataloader_tfrecord.py)
4. [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) - human speech, extracted from YouTube interview videos
  - [ ] Pytorch loader
  - [ ] TFRecords loader
5. [GTZAN](http://marsyasweb.appspot.com/download/data_sets/) - audio tracks from a variety of sources annotated with genre class
  - [x] [GTZAN Downloader](https://github.com/juliagusak/dataloaders/blob/master/GTZAN/gtzan_dataset.py)
  - [x] [PyTorch DataSet](https://github.com/juliagusak/dataloaders/blob/master/GTZAN/torch/gtzan_dataset.py)
6. CallCenter - audio tracks with human and non-human speech
  - [x] [PyTorch DataSet](https://github.com/juliagusak/dataloaders/blob/master/GTZAN/torch/callcenter_dataset.py)
  
For validation we frequently use the following scheme: 
1. Read 10 random crops from a file;
2. Predict a class for each crop;
3. Averaging results.

For this scheme we've done additional DataLoaders for PyTorch:

  - [DataLoader for ESC, GTZAN, LibriSpeech](https://github.com/juliagusak/dataloaders/blob/master/mics/data_loader.py)
  - [DataLoader for LibriSpeech from TfRecords](https://github.com/juliagusak/dataloaders/blob/master/librispeech/tfrecord_dataloader.py)  
  - [DataLoaders for NSynth](https://github.com/juliagusak/dataloaders/blob/master/librispeech/tfrecord/tfrecord_dataloader.py)
