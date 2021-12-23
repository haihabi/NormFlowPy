from sklearn.datasets import make_moons


def generate_moons_dataset(dataset_size):
    x,y= make_moons(n_samples=dataset_size, shuffle=True, noise=0.05)
    return x.astype("float32"),y.astype("float32")
