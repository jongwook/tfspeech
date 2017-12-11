"""
Creates an 8:1:1 train:validation:test split using the hash scheme provided by Kaggle
"""

import os
import re
import numpy as np
import hashlib
from tqdm import tqdm
from librosa.util import frame, normalize
import gzip
import scipy.io.wavfile as wavfile
from pathlib import Path

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, validation_percentage=10, testing_percentage=10):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

    Returns:
    String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name).encode('utf-8')
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'test'
    else:
        result = 'train'
    return result

def preprocess_dataset(audio_path="data/train/audio", output_path="data/preprocessed"):
    """
    reads audio files in `audio_path` and split them into 8:1:1 split,
    and store them into `{train|validation|test}/{label}.npy`, under `output_path`.
    each file represents a N-by-16000 matrix where each row is the 1-second audio
    """
    for subdir in ["train", "validation", "test"]:
        Path(os.path.join(output_path, subdir)).mkdir(parents=True, exist_ok=True)

    for label in tqdm(os.listdir(audio_path), desc="Looping over each label"):
        path = os.path.join(audio_path, label)
        if not os.path.isdir(path):
            continue

        data = {
            'train': [],
            'validation': [],
            'test': []
        }

        files = [f for f in os.listdir(path) if f.endswith(".wav")]

        for filename in tqdm(files, desc="Loading %s" % label):
            fs, w = wavfile.read(os.path.join(path, filename))
            assert fs == 16000
            if label == "_background_noise_":
                audio_length = int(len(w) / 16000)
                pivots = [int(audio_length * p) * 16000 for p in [0.8, 0.9, 1.0]]
                train = w[:pivots[0]]
                validation = w[pivots[0]:pivots[1]]
                test = w[pivots[1]:pivots[2]]

                data['train'].append(frame(train, frame_length=16000, hop_length=2000).transpose())
                data['validation'].append(frame(validation, frame_length=16000, hop_length=2000).transpose())
                data['test'].append(frame(test, frame_length=16000, hop_length=2000).transpose())
            else:
                assert len(w.shape) == 1 and len(w) <= 16000

                if len(w) < 16000:
                    pad = 16000 - len(w)
                    pad_left = pad // 2
                    pad_right = pad - pad_left
                    w = np.pad(w, (pad_left, pad_right), mode='constant')

                assert len(w) == 16000

                split = which_set(filename)
                data[split].append(w)

        for split in data.keys():
            matrix = np.vstack(data[split])
            assert matrix.shape[0] > 0 and matrix.shape[1] == 16000
            path = os.path.join(output_path, split, label + ".npy")
            with open(path, 'wb') as f:
                np.save(f, matrix)

def preprocess_test_dataset(audio_path='data/test/audio', output_path='data/preprocessed'):
    files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
    data = []

    for filename in tqdm(files, desc="Loading files"):
        fs, w = wavfile.read(os.path.join(audio_path, filename))
        assert fs == 16000
        assert len(w.shape) == 1 and len(w) <= 16000

        if len(w) < 16000:
            pad = 16000 - len(w)
            pad_left = pad // 2
            pad_right = pad - pad_left
            w = np.pad(w, (pad_left, pad_right), mode='constant')

        assert len(w) == 16000
        data.append(w)

    matrix = np.vstack(data)
    assert matrix.shape[0] > 0 and matrix.shape[1] == 16000
    path = os.path.join(output_path, 'test.npy')
    with open(path, 'wb') as f:
        np.save(f, matrix)

if __name__ == "__main__":
    preprocess_dataset()
