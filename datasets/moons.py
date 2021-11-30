from datasets.numpy_dataset import NumpyDataset


def generate_moons_dataset(dataset_size, labeled: bool = False):
    self.x, self.y = make_moons(n_samples=dataset_size, shuffle=True, noise=0.05)
    return NumpyDataset(x)
