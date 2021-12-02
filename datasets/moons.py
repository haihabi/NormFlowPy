from sklearn.datasets import make_moons


def generate_moons_dataset(dataset_size):
    return make_moons(n_samples=dataset_size, shuffle=True, noise=0.05)
