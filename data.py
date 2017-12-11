import os
import numpy as np
from tqdm import tqdm

def load(data_path='data/preprocessed'):
    """
    loads the preprocessed files in '.npy' format and returns a 3-element tuple,

    ((train_data, train_label), (validation_data, validation_label), (test_data, test_label))

    where each data matrix is N-by-16000, and each label matrix is of size N.
    """

    result = []
    last_files = None

    for split in ['train', 'validation', 'test']:
        path = os.path.join(data_path, split)
        files = [f for f in os.listdir(path) if f.endswith('.npy')]
        files.sort()

        if last_files is not None and not [x == y for x, y in zip(last_files, files)]:
            raise Exception("split %s has a different set of data files" % split)

        matrices = []
        labels = []

        # label ranges betwen [0-30], where 0 is _background_noise_
        for i in tqdm(range(len(files)), desc='Loading %s files' % split):
            matrix = np.load(os.path.join(path, files[i]))
            label = np.zeros((matrix.shape[0], len(files)))
            label[:, i] = 1

            matrices.append(matrix)
            labels.append(label)

        result.append((np.vstack(matrices), np.vstack(labels)))

    return tuple(result)
