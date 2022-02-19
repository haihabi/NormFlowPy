from experiments.datasets.moons import generate_moons_dataset
from experiments.datasets.numpy_dataset import NumpyDataset
from enum import Enum


class DatasetType(Enum):
    MOONS = 0


def get_dataset(dataset_type: DatasetType, dataset_size: int, labeled: bool = False):
    if dataset_type == DatasetType.MOONS:
        x, y = generate_moons_dataset(dataset_size)
    else:
        raise Exception("Unknown dataset type")
    if labeled:
        return NumpyDataset(x, y)
    return NumpyDataset(x)
