"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from dataset.dataset_classes.dataset_gen import DatasetGen


class CodiDataset(DatasetGen):
    """
    Class for the Dataset used to train CoDi.
    Also stores the test and validation sets.
    """
    def __init__(self, codi_labelled):
        self.dataset = codi_labelled

    def get_dataset(self):
        return self.dataset
