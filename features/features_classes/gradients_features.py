"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from .feature import Feature
import numpy as np

class GradientFeature(Feature):
    r"""
    A Feature that extracts metrics from gradients of samples
    :param
    n_classes : number of classes in the classification task
    :attr
    n_classes : number of classes in the classification task
    gradients : list of gradients
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.gradients = []
        self.grad_variance = []
        self.grad_min_margin = []
        self.grad_min = []

    def augment(self, grad):
        """
        Computes the metrics for the current samples and saves them in class attributes
        :param grad: array-like of shape (batch_size, n_labels)
        :return: None
        """
        self.gradients.append(grad)
        self.grad_variance = grad.var()
        self.grad_min = grad.min()
        grad = np.sort(grad)
        self.grad_min_margin.append(grad[0][1] - grad[0][0])


    def get_features(self):
        """
        A getter for the class attributes
        :return: A 4-tuple of arrays of the class attributes
        (gradients, grad_margin, grad_variance, min_gradient) each
        of shape (n_samples, n_classes)
        """
        return np.array(self.gradients), \
            np.array(self.grad_variance), \
            np.array(self.grad_min_margin), \
            np.array(self.grad_min)

    def get_gradients(self):
        """
         A getter for gradients
        :return: array-like of shape (n_samples, n_classes)
        """
        return np.array(self.gradients)


