"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from abc import abstractmethod
from torch import nn


class InferenceModel(nn.Module):
    """
    An abstract class for the implementation of the Inference Model for the Inference phase.
    The children classes should be able to load the model from yaml, train it, infer labels on the unlabelled dataset
    and process the labelled dataset 2.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_from_yaml(self, yaml_file, encoder):
        pass

    @abstractmethod
    def train_model(self, labelled_dataset1, labelled_test1, validation_set1):
        pass

    @abstractmethod
    def infer_labels(self):
        pass

    @abstractmethod
    def process_codi_labelled_dataset(self):
        pass
