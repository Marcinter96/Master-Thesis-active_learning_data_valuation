"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from dataset.dataset_classes.dataset_gen import DatasetGen


class InferenceDataset(DatasetGen):
    """
    Class for the Labelled Dataset 1.
    Also stores the test and validation sets.
    """
    def __init__(self, labelled_dataset1, labelled_test1, validation_set1):
        self.dataset = labelled_dataset1
        self.test = labelled_test1
        self.validation = validation_set1

    def get_dataset(self):
        return self.dataset

    def get_pack(self):
        return self.dataset, self.test, self.validation
