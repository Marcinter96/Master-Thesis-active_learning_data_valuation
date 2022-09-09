"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from .feature import Feature
from scipy.stats import entropy
import numpy as np


class KLDivergence(Feature):
    r"""
    A feature that computes the KL divergence between the
    logits of each data points given by a classifier mean logits
    for each label and the mean of these logits for each label

    ----------
    mean_logits : array-like of shape (n_classes, n_classes) is the mean of the logits of datapoints
    having the same label. First dimension should be labels, second should be the mean logit for
    this label

    Attributes
    ----------
    mean_logits: ' '
    """

    def __init__(self, mean_logits):
        self.mean_logits = mean_logits

    def augment(self, logits):
        """
        Performs the data augmentation.

        Computes the KL divergence between the parameter logits and
        the attribute mean_logits
        :param
        logits: array-like of shape (n_classes, n_samples)
        :return:
        C : array-like of shape (n_classes, n_samples)
        """
        return np.array([entropy(logits,
                                 np.repeat(mean_logit[..., np.newaxis], logits.shape[1], axis=1), base=2)
                         for mean_logit in self.mean_logits])
