from abc import abstractmethod


class ConfidenceDiscriminator:
    r"""
    A classifier interface for the different accept/reject classifier
    """
    @abstractmethod
    def call(self, inputs):
        pass
