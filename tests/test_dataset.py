import unittest
import datasets


class TestDataset(unittest.TestCase):
    def general_test_something(self, dataset_type):
        ds = datasets.get_dataset(dataset_type, 10)
        self.assertTrue(len(ds) == 10)

    def test_moon_dataset(self):
        self.general_test_something(datasets.DatasetType.MOONS)


if __name__ == '__main__':
    unittest.main()
