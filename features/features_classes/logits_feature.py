"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from .feature import Feature
import numpy as np
from scipy.stats import entropy
MIN_LOGIT = 1e-2


class LogitsFeature(Feature):
    r"""
    A Feature that extracts four different metrics from logits of samples
    :param
    n_classes : number of classes in the classification task
    sparse : whether the logits for the task are given as array
        of alternating values index-probability (ex: next word prediction)
    sequence: whether logits come as a sequence (ex: next word prediction)
    :attr
    n_classes : number of classes in the classification task
    sparse : whether the logits for the task are given as array
        of alternating values index-probability (ex: next word prediction)
    logits : list of logits
    confidence_margin : the difference between the two
        maximal class probabilities for each sample
    confidence_margin : the ratio between the two
        maximal class probabilities for each sample
    entropy : the entropy of the logits
    """

    def __init__(self, n_classes=6, sparse=False, sequence=False):
        self.n_classes = n_classes
        self.sparse = sparse
        self.sequence = sequence
        self.logits = []
        self.confidence_margin = []
        self.confidence_ratio = []
        self.entropy = []

    def augment(self, logit):
        """
        Computes the metrics for the current samples and saves them in class attributes
        :param logit: array-like of shape (batch_size, n_labels)
        :return: None
        """
        self.logits.append(logit)
        if self.sparse:
            logit = logit[:, 1::2]
        if self.sequence:
            first_logit = logit[:, :, 0]
            second_logit = logit[:, :, 1]
            margin = first_logit - second_logit
            ratio = first_logit / second_logit
            ratio[second_logit < MIN_LOGIT] = -1
            self.confidence_margin.append(margin)
            self.confidence_ratio.append(ratio)
            self.entropy = np.ones(np.shape(self.confidence_ratio))
        else:
            logit = np.sort(logit, axis=-1)
            self.confidence_margin.append(logit[:, -1] - logit[:, -2])

            second_logit = logit[:, -2]
            ratio = logit[:, -1]/logit[:, -2]
            ratio[second_logit < MIN_LOGIT] = -1
            self.confidence_ratio.append(ratio)

            self.entropy.append(entropy(logit.T, base=self.n_classes))

    def get_features(self):
        """
        A getter for the class attributes
        :return: A 4-tuple of arrays of the class attributes
        (logits, confidence_margin, confidence_ratio, entropy) each
        of shape (n_samples, n_classes)
        """
        return np.array(self.logits), \
            np.array(self.confidence_margin), \
            np.array(self.confidence_ratio), \
            np.array(self.entropy)

    def get_logits(self):
        """
        A getter for logits
        :return: array-like of shape (n_samples, n_classes)
        """
        return np.array(self.logits)

    def get_class_mean_logits(self):
        """
        Computes the mean logit per class
        :return:
        C : array-like of shape (n_classes, n_classes) is the mean of the logits of datapoints
        having the same label. First dimension should be labels, second should be the mean logit for
        this label
        """
        labels = np.argmax(self.logits, axis=1)
        mean_logits = []
        for i in range(self.n_classes):
            mean_logits.append(self.logits[labels == i].mean())
        return np.array(mean_logits)
