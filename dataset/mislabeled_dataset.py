"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

import seaborn as sns


class MislabeledDataset:
    """
    This class is used to simulate the setup of spotting mislabeled points in a mislabeled dataset.
    At init, original labels are stored then labels are being flipped.
    This class is being used to simulate a setup where given labels are likely to be incorrect.
    It also provides plotting functions to show the results of confidence scoring methods.
    """

    def __init__(self, original_dataset, mislabel_ratio):
        """
        Initializes the MislabeledDataset object then stores the true labels of the original dataset and flips a certain
        portion of labels in this dataset.
        :param original_dataset: Given dataset that we want to mislabel.
        :param mislabel_ratio: Percentage of labels that will be flipped.
        """
        self.dataset = original_dataset
        self.mislabel_ratio = mislabel_ratio

        self.true_labels = None
        self.unknown_labels = None
        self.corresponding_logits = None
        self.mislabel_indices = None

        self.__extract_labels()
        self.__flip_labels()

    def __extract_labels(self):
        """
        Helper function to store the original true labels before flipping them.
        """
        self.true_labels = np.fromiter(self.dataset.map(lambda data_point, label: label).as_numpy_iterator(),
                                       int)

    def __flip_labels(self):
        """
        Method to flip some of the labels of the original dataset according to the given mislabel_ratio percentage.
        """
        amount_to_flip = int(self.mislabel_ratio * self.true_labels.shape[0])
        self.mislabel_indices = np.random.choice(np.arange(self.true_labels.shape[0]), amount_to_flip, replace=False)

        random_shift = np.zeros(self.true_labels.shape[0])
        random_shift[self.mislabel_indices] = 1

        # This operation is only good for IMDB as we assume that the number of classes is 2
        self.unknown_labels = (self.true_labels + random_shift).astype(int) % 2

        print('Labels have been flipped.')

    def get_mislabeled_dataset(self):
        return self.dataset

    def extract_corresponding_logits(self, prediction_logits):
        """
        Method for the corresponding logits scoring method on a mislabel dataset.
        :param prediction_logits: prediction logits given by the inference model.
        :return: the corresponding logits score
        """
        self.corresponding_logits = np.array([prediction_logits[ind, which_one] for ind, which_one in
                                              enumerate(self.unknown_labels)])

        return self.corresponding_logits

    @staticmethod
    def normalized_array(n):
        """
        Helper function used for plotting the mislabel spotting curve.
        :param n: length of the output array
        :return: output array used for plotting the score curve.
        """
        return np.arange(n)/(n-1)

    @staticmethod
    def score(mislabelled_indices, sorted_values):
        """
        Computes the sorted values scores.
        This is the method used to plot the 'Found mislabelled vs. data explored'.
        It counts the number of mislabelled points found when checking the points corresponding to their influence
        values scores.

        :param mislabelled_indices: the list of indices corresponding to the mislabelled data points
        :param sorted_values: the list of indices corresponding to the sorted influence values starting with the
        :return: the scores corresponding to the fraction of mislabelled data found according to the fraction of total
        data checked.
        """
        mislabelled_indices = set(mislabelled_indices)

        scores = []
        current_score = 0

        for val in sorted_values:
            if val in mislabelled_indices:
                current_score += 1
            scores.append(current_score)

        return scores

    def plot_method(self, title):
        """
        Plotting method for mislabel spotting.
        This plots the curve of fraction of mislabeled points spotted vs. amount of data examined.
        This sorts the points according to their scores, and starts examining the points with the lowest scores first.
        Also prints the Area Under Curve of the obtained curve.
        :param title: Title of the plot
        """
        plt.figure(figsize=(10, 7))

        sorted_values = sorted([(ind, ele) for ind, ele in enumerate(self.corresponding_logits)], key=lambda x: x[1])
        sorted_indices = [a for a, _ in sorted_values]

        sv_scores = MislabeledDataset.score(self.mislabel_indices, sorted_indices)
        sv_scores = np.array(sv_scores) / max(sv_scores)

        auc_score = auc(MislabeledDataset.normalized_array(len(sv_scores)), sv_scores)

        if auc_score < 0.5:
            sorted_values = sorted([(ind, -ele) for ind, ele in enumerate(self.corresponding_logits)],
                                   key=lambda x: x[1])
            sorted_indices = [a for a, _ in sorted_values]

            sv_scores = MislabeledDataset.score(self.mislabel_indices, sorted_indices)
            sv_scores = np.array(sv_scores) / max(sv_scores)

            auc_score = auc(MislabeledDataset.normalized_array(len(sv_scores)), sv_scores)

        x_plot = MislabeledDataset.normalized_array(len(sv_scores))
        plt.plot(x_plot, sv_scores, label='Corresponding logit')

        plt.legend()
        plt.title('Detection from training set ({}% mislabeled)'.format(self.mislabel_ratio*100))
        plt.xlabel('Fraction of data examined')
        plt.ylabel('Fraction of mislabelled data found')

        print()
        print("AUC for the method {} with ratio {}: {}".format(title, self.mislabel_ratio, auc_score))

        # Plotting the performances when checking data randomly
        plt.plot(x_plot, np.linspace(0, 1, len(x_plot)), color='red')
        plt.text(0.5, 0.45, "Results with random check", rotation=35)

        plt.show()

    def plot_score_distribution(self):
        """
        Plotting method for the distribution of a score. In this case it is used for the corresponding logits score.
        """
        plt.figure(figsize=(10, 7))
        sns.distplot(self.corresponding_logits, kde=False, hist=True, label='Prediction logit score distribution')

        plt.show()

    @staticmethod
    def plot_distribution_labelled2(true_indices_labelled2, confidence_score, title, x_label):
        """
        Method used to plot the distribution of a given score for correct points and mislabeled points.
        Both distributions of the scores will be plotted comparatively in normalized histograms.
        :param true_indices_labelled2: the indices of the correct points. Allows this method to plot the two
        distributions
            for correct and mislabelled points.
        :param confidence_score: the given score that we want to plot the distribution.
        :param title: Title of the plot
        :param x_label: Label of the x axis (likely to be the name of the given confidence score)
        """
        indices_correct_points = true_indices_labelled2.astype(bool)
        indices_incorrect_points = np.invert(indices_correct_points)

        correct_scores = confidence_score[indices_correct_points]
        incorrect_scores = confidence_score[indices_incorrect_points]

        plt.figure(figsize=(10, 7))
        plt.hist([correct_scores, incorrect_scores], color=['b', 'r'], bins='auto', density=True)
        plt.legend(['Correct points', 'Mislabeled points'])
        plt.xlabel(x_label)
        plt.ylabel('Number of points')
        plt.title(title)
        plt.show()
