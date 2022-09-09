"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from dataset.dataset_classes.dataset_gen import DatasetGen


class UnlabelledDataset(DatasetGen):
    """
    Class for the Unlabelled Dataset.
    """
    def __init__(self, unlabelled_dataset):
        self.dataset = unlabelled_dataset

    def get_dataset(self):
        return self.dataset
