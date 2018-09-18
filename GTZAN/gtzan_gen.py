import argparse
import glob
import os
import subprocess
from random import shuffle

import librosa
import numpy as np

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from misc.utils import LabelsEncoder

GTZAN_SPEECH_URL = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
DEFAULT_BIT_RATE = 22050

TAR_FILE = "genres.tar.gz"
FOLDER_NAME = TAR_FILE[:-7]
TRAIN_SUFFIX = "train.npz"
TEST_SUFFIX = "test.npz"
VAL_SUFFIX = "val.npz"


def parse_args():
    parser = argparse.ArgumentParser(description='GTZAN')

    # General settings
    parser.add_argument('--path', required=True, help="Where to store results")
    parser.add_argument('--train', default=1.0, type=float, help="What fraction take for training")
    parser.add_argument('--val', default=0.0, type=float, help="Where fraction take for testing")
    parser.add_argument('--force_download', action='store_true', help="Force downloading from website.")
    parser.add_argument('--force_extraction', action='store_true', help="Forcing extraction from tar.gz file.")
    parser.add_argument('--force_npz', action='store_true', help="Forcing convertation to wav")
    parser.add_argument('--force_h5py', action='store_true', help="Forcing storing to h5py_utils")
    parser.add_argument('--sr', default=16000, type=int, help="Sample rate for wav. Default is 16kHz")
    parser.add_argument('--n_jobs', default=4, type=int, help="Number of threads for reading audio samples")

    return parser.parse_args()


def save_npz(X, y, z, save_to):
    data = {"X": X, "y": y, "label_name": z}
    np.savez(save_to, **data)


def read_file(file_name, sr, verbose=0):
    if verbose:
        print("Read file:", file_name)
    label = file_name.split('/')[-2]
    audio, sr = librosa.core.load(file_name, sr, res_type='kaiser_fast')
    return audio, label


if __name__ == "__main__":
    opt = parse_args()
    tar_gz_path = os.path.join(opt.path, TAR_FILE)
    extracted_path = os.path.join(opt.path, FOLDER_NAME)
    if not os.path.exists(tar_gz_path) or opt.force_download:
        if opt.force_download and os.path.exists(tar_gz_path):
            print("Force download. {} file will me replaced.".format(tar_gz_path))
            os.remove(tar_gz_path)

        print("Download *.tar.gz file to", tar_gz_path)
        subprocess.run("wget {} -P {}".format(GTZAN_SPEECH_URL, opt.path), shell=True, check=True)
    else:
        print("The dataset has been already downloaded to {}".format(tar_gz_path))

    if not os.path.exists(extracted_path):
        print("Extract data to", extracted_path)
        subprocess.run("tar xvzf {} -C {}".format(tar_gz_path, opt.path), shell=True, check=True)
    else:
        print("The dataset has been already extracted to {}".format(extracted_path))

    print("Read in memory")
    X = []
    y = []

    file_names = glob.glob(extracted_path + '/**/*.au')
    shuffle(file_names)
    file_names = file_names
    result = Parallel(n_jobs=opt.n_jobs, verbose=0)(delayed(read_file)(file_name, opt.sr, 1) for file_name in file_names)
    X, y = zip(*result)

    X = np.array(X)
    y = np.array(y)

    encoder = LabelsEncoder(y)
    z = encoder(y)

    print("Finish")

    if opt.train < 1.0:
        X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size=1-opt.train)
    else:
        X_train, y_train, z_train = X, y, z

    save_npz(X_train, z_train, y_train,
             os.path.join(opt.path, FOLDER_NAME) + "{}_{}".format(int(opt.sr // 1000), TRAIN_SUFFIX))

    if opt.train < 1.0:
        if opt.val == 0:
            save_npz(X_test, z_test, y_test,
                     os.path.join(opt.path, FOLDER_NAME) + "{}_{}".format(int(opt.sr // 1000), TEST_SUFFIX))
        else:
            X_test, X_val, y_test, y_val, z_test, z_val = train_test_split(X_test, y_test, z_test, test_size=opt.val)
            save_npz(X_test, z_test, y_test,
                     os.path.join(opt.path, FOLDER_NAME) + "{}_{}".format(int(opt.sr // 1000), TEST_SUFFIX))
            save_npz(X_val, z_val, y_val,
                     os.path.join(opt.path, FOLDER_NAME) + "{}_{}".format(int(opt.sr // 1000), VAL_SUFFIX))

