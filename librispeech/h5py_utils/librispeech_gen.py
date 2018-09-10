import argparse
import h5py
import numpy as np
import os
import subprocess
import wavio

from glob import glob
from tqdm import tqdm
from random import shuffle

from utils import UTTERANCE, CHAPTER, SPEAKER, SOUND

LIBRI_SPEECH_URL = "http://www.openslr.org/12/"
EXTRACTED_FOLDER = "LibriSpeech"


def parse_args():
    parser = argparse.ArgumentParser(description='LibriSpeech')

    # General settings
    parser.add_argument('--dataset',
                        required=True,
                        help="The name of a particular dataset from {}".format(LIBRI_SPEECH_URL))
    parser.add_argument('--url',
                        default=LIBRI_SPEECH_URL,
                        help="Where datasets are stored. Default: {}".format(LIBRI_SPEECH_URL))
    parser.add_argument('--path', required=True, help="Where to store results")
    parser.add_argument('--force_download', default=False, help="Force downloading from website.")
    parser.add_argument('--force_extraction', default=False, help="Forcing extraction from tar.gz file.")
    parser.add_argument('--force_convert', default=False, help="Forcing convertation to wav")
    parser.add_argument('--force_h5py', default=False, help="Forcing storing to h5py_utils")
    parser.add_argument('--sr', default=16000,  help="Sample rate for wav. Default is 16kHz")
    parser.add_argument('--wav_dir', default=EXTRACTED_FOLDER+"Wav",  help="Where to store wav files")
    parser.add_argument('--rm_flac', default=True, help="Remove or not folder with flac files")
    parser.add_argument('--take_random', default=None, type=int, help="Take N random wav files for storing in h5py_utils")

    return parser.parse_args()


if __name__=="__main__":
    opt = parse_args()

    # Download tar
    data_url = os.path.join(opt.url, opt.dataset)
    tar_path = os.path.join(opt.path, opt.dataset)
    extraction_path = os.path.join(opt.path, EXTRACTED_FOLDER, opt.dataset[:-7])
    wav_path = os.path.join(opt.path, opt.dataset[:-7] + "_wav" + str(opt.sr // 1000))
    dataset_path = os.path.join(opt.path, opt.dataset[:-7])+'.hdf5'

    if opt.force_h5py:
        print("Force h5py_utils creation. {} file will me replaced.".format(dataset_path))
        subprocess.run("rm -rf {}".format(dataset_path), shell=True, check=True)

    if os.path.exists(dataset_path) and not (opt.force_download or opt.force_extraction or opt.force_convert):
        print('Dataset is already downloaded and prepared')
        exit()

    # rm folders
    if opt.force_download:
        if opt.force_download:
            print("Force download. {} file will me replaced.".format(tar_path))
            os.remove(tar_path)

    if opt.force_extraction:
        print("Force extraction. {} file will me replaced.".format(extraction_path))
        subprocess.run("rm -rf {}".format(extraction_path), shell=True, check=True)

    if opt.force_convert:
        print("Force extraction. {} file will me replaced.".format(wav_path))
        subprocess.run("rm -rf {}".format(wav_path), shell=True, check=True)

    if not os.path.exists(tar_path) or opt.force_download:
        print("Download tar.gz")
        subprocess.run("wget {} -P {}".format(data_url, opt.path), shell=True, check=True)
    else:
        print("Dataset has already downloaded")

    # Extract tar
    if (not os.path.exists(extraction_path) and not os.path.exists(wav_path)) or opt.force_extraction:
        print("Extraction path:", extraction_path)
        subprocess.run("tar xvzf {} -C {}".format(tar_path, opt.path), shell=True, check=True)
    else:
        print("Dataset has already extracted")

    # Convert to wav
    wav_path = os.path.join(opt.path, opt.dataset[:-7] + "_wav" + str(opt.sr//1000))
    print("wav_path", wav_path)
    if not os.path.exists(wav_path) or opt.force_convert or opt.force_extraction:
        os.mkdir(wav_path)
        flacs = glob('{}/**/*.flac'.format(extraction_path), recursive=True)
        for flac in flacs:
            wav_file = os.path.join(wav_path, flac.split("/")[-1][:-5] + '.wav')
            subprocess.run('ffmpeg -i {} {} -ar {}'.format(flac, wav_file, opt.sr), shell=True, check=True)
    else:
        print("Dataset has already converted to wav with sr {}".format(opt.sr))

    if opt.rm_flac and os.path.exists(extraction_path):
        print("Flac folder {} will be removed".format(extraction_path))
        subprocess.run("rm -rf {}".format(extraction_path), shell=True, check=True)

    print("Convertation to wav is finished")

    if not os.path.exists(dataset_path):
        print("Packing into {} file".format(dataset_path))
        wav_files = os.listdir(wav_path)
        shuffle(wav_files)
        if opt.take_random is not None:
            wav_files = wav_files[:opt.take_random]

        data_len = len(wav_files)
        f = h5py.File(dataset_path, 'w')

        dt = h5py.special_dtype(vlen=np.float32)
        sound = f.create_dataset(SOUND, (data_len, ), dtype=dt)
        speaker = f.create_dataset(SPEAKER, (data_len,), dtype=np.int)
        chapter = f.create_dataset(CHAPTER, (data_len,), dtype=np.int)
        utterance = f.create_dataset(UTTERANCE, (data_len,), dtype=np.int)

        for i, wav_file in tqdm(enumerate(wav_files), total=data_len):
            file_name = wav_file.split("/")[-1][:-4]
            sound_wav = wavio.read(os.path.join(wav_path, wav_file)).data.T[0]
            speaker_id, chapter_id, utterance_id = map(int, file_name.split("-"))
            sound[i] = sound_wav
            speaker[i], chapter[i], utterance[i] = speaker_id, chapter_id, utterance_id

        f.flush()
        f.close()
    else:
        print("{} file has been already prepared.".format(dataset_path))

