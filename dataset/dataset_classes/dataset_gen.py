"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from abc import abstractmethod
from abc import ABC


class DatasetGen(ABC):
    """
    An abstract class for the different dataset classes
    """
    @abstractmethod
    def get_dataset(self):
        pass
