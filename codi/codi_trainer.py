"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from abc import abstractmethod


class CoDiTrainer:
    r"""
    A classifier interface for the different accept/reject classifier
    """
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def create_labelled_dataset(self):
        pass

    @abstractmethod
    def create_unlabelled_dataset(self):
        pass

    @abstractmethod
    def test(self):
        pass
