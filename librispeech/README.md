LibriSpeech Readers
============================
To generate h5py file with [LibriSpeech](http://www.openslr.org/12/) data evaluate the following cmd:
    
    python librispeech_gen.py --dataset train-clean-100.tar.gz --path ./librispeech --force_h5py True
    
For more options do:

    python librispeech_gen.py -h
    
To create PyTorch dataset from the *.h5py file please use the class *LibriSpeechH5py* from *h5py_reader.py*.
To evaluate test data loader use *LibriSpeechH5pyTestDataLoader* from  *data_loader.py*. 
You can find usage examples in *h5py_reader.py*, *data_loader.py*. 
 