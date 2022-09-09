"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from abc import abstractmethod


class Feature:
    r"""
    A Feature interface for the different feature augmentations
    """
    @abstractmethod
    def augment(self, data):
        pass
