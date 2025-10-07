import numpy as np


def to_statistics(sequence):
    array = np.array(sequence)
    return np.mean(array), np.std(array), np.min(array), np.max(array)