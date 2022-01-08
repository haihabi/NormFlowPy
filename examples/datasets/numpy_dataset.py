import numpy as np
from torch.utils.data.dataset import Dataset


class NumpyDataset(Dataset):
    def __init__(self, data, label=None):
        self.data = data
        self.label = label
        self.is_labeled = label is not None
        self.n = len(data)

    def __getitem__(self, index):
        d = self.data[index]
        if not self.is_labeled:
            return d
        l = self.label[index]
        return d, l

    def __len__(self):
        return self.n

    def dim(self) -> int:
        return int(np.prod(self.__getitem__(0).shape))
