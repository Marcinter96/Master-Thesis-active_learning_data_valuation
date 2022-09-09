"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from .feature import Feature
import numpy as np

class InfluenceFeature(Feature):
    r"""
    A Feature that extracts metrics from influence score of samples
    :param
    n_classes : number of classes in the classification task
    :attr
    n_classes : number of classes in the classification task
    inf : list of influence_score
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.influence = []
        self.influence_variance = []
        self.influence_min_margin = []
        self.influence_min = []

    def augment(self, inf):
        """
        Computes the metrics for the current samples and saves them in class attributes
        :param inf: array-like of shape (batch_size, n_labels)
        :return: None
        """
        self.influence.append(inf)
        self.influence_variance = inf.var()
        self.influence_min = inf.min()
        inf = np.sort(inf)
        self.influence_min_margin.append(inf[0][1] - inf[0][0])


    def get_features(self):
        """
        A getter for the class attributes
        :return: A 4-tuple of arrays of the class attributes
        (inf, inf_margin, inf_variance, min_inf) each
        of shape (n_samples, n_classes)
        """
        return np.array(self.influence), \
            np.array(self.influence_variance), \
            np.array(self.influence_min_margin), \
            np.array(self.influence_min)

    def get_influence(self):
        """
         A getter for influence score
        :return: array-like of shape (n_samples, n_classes)
        """
        return np.array(self.influence)


