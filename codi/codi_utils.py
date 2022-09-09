"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from features.features_classes.logits_feature import LogitsFeature


def generate_plots(values):
    """
    Generates plots of train and validation losses and accuracies
    """
    if not os.path.exists('codi/figures'):
        os.makedirs('codi/figures')
    plt.figure()
    for i in range(0, len(values)):
        plt.plot(values[i])
    plt.legend(('train_acc', 'train_loss', 'val_loss', 'val_acc'))
    plt.xlabel('# epochs')
    plt.ylabel('metric value')
    plt.savefig('codi/figures/metrics', format='pdf')


def plot_precision_threshold_tradeoff(pred, label):
    """
    Generates plots of precision of class 1 vs threshold of sorted points taken.
    """
    if not os.path.exists('codi/figures'):
        os.makedirs('codi/figures')
    print(classification_report(label, np.array(pred >= 0.5)))
    plt.figure()
    flat_preds = [item for sublist in pred.tolist() for item in sublist]
    pred_sorted = np.sort(np.array(flat_preds))[::-1]
    pred_sorted_idx = np.argsort(np.array(flat_preds))[::-1]
    true_sorted = label[pred_sorted_idx]
    precision_score = []
    for thresh in np.logspace(0, 2, 100)[:-1]:
        to_keep = np.floor((thresh * len(pred_sorted))/100)
        pred_artificial = np.array([pred_sorted >= pred_sorted[int(to_keep)]])
        precision_score.append(precision_recall_fscore_support(true_sorted,
                                                               pred_artificial[0])[0][1])
    plt.plot(np.logspace(0, 2, 100)[:-1], precision_score)
    plt.xlabel('Threshold')
    plt.ylabel('Precision of class 1')
    plt.title('Precision-Threshold tradeoff of class 1')
    plt.savefig('codi/figures/precision_threshold_tradeoff', format='pdf')


def filter_nlp_datapoints(iterator, filtering):
    """
    Adds an attribute 'is_accepted' to the iterator as a boolean list of length nb_thresh.
    """
    if filtering['exp'] is False:
        thresholds = np.array(filtering['thresh'])
    else:
        if filtering['by_percentage'] is True:
            thresholds = np.linspace(10, 100, filtering['nb_thresh'])/100
        else:
            thresholds = 1.08 - np.logspace(1, 1.995, filtering['nb_thresh'])/100

    scores = np.ones((0, 1))

    for example_batch, _ in iterator:
        score_batch = [example.score for example in example_batch]
        scores = np.append(scores, score_batch, axis=0)

    if filtering['by_percentage'] is True:
        quantiles = np.quantile(scores, 1-thresholds)

    for example_batch, _ in iterator:
        for _, example in enumerate(example_batch):
            if filtering['in_between'] is False:
                if filtering['by_percentage'] is True:
                    example.is_accepted = quantiles >= example.score
                else:
                    example.is_accepted = thresholds >= example.score
            else:
                quantiles = np.quantile(scores, 1 - thresholds)
                augmented_quantiles = np.append([1], quantiles)
                augmented_quantiles[-1] = 0
                prev_thresh = augmented_quantiles[0]
                for i in range(1, len(augmented_quantiles)):
                    if (example.score <= prev_thresh and example.score >= augmented_quantiles[i]):
                        example.is_accepted = [0]*filtering['nb_thresh']
                        example.is_accepted[i-1] = 1
                    prev_thresh = augmented_quantiles[i]


def save_ids(ids):
    """
    Saves IDS of points to retrain
    """
    if not os.path.exists('codi/outputs'):
        os.makedirs('codi/outputs')
    np.save('codi/outputs/ids_trust', ids[0])
    np.save('codi/outputs/ids_no_trust', ids[1])


def generate_box_plot(X, feature_dict, list_ids):
    """
    Plots boxplots of output of crossvalidation
    """
    if not os.path.exists('codi/figures'):
        os.makedirs('codi/figures')
    # Change list in dict to tuples.
    feature_dict = dict([a, tuple(x)] for a, x in feature_dict.items())
    # Invert values and keys in the dictionary
    inv_map = {v: k for k, v in feature_dict.items()}
    xticks_all = []
    plt.figure(figsize=[12, 7])

    for i in range(len(X.T)):
        xticks = []
        for j in range(0, len(list_ids[i])):
            if type(list_ids[i][j]) == list:
                xticks.append(inv_map[tuple(list_ids[i][j])])
            else:
                xticks.append(inv_map[list_ids[i][j]])
        xticks_all.append(xticks)
    plt.boxplot(X.T.tolist())
    plt.xlabel('Model features')
    plt.xticks(np.arange(1, len(X.T)+1), xticks_all, rotation=18, fontsize=5)
    plt.xlabel('Model features', fontsize=12)
    plt.ylabel('Precision of class 1')
    plt.savefig('codi/figures/precision_boxplot', format='pdf')


def plot_combination_accuracy(feature_dict, accs, list_ids, figname):
    """
    This function handles the plots of accuracy for each model after combination
    selection.
    """
    if not os.path.exists('codi/figures'):
        os.makedirs('codi/figures')
    # Change list in dict to tuples.
    feature_dict = dict([a, tuple(x)] for a, x in feature_dict.items())
    # Invert values and keys in the dictionary
    inv_map = {v: k for k, v in feature_dict.items()}
    xticks_all = []
    plt.figure(figsize=[12, 7])

    x = np.arange(0, len(accs))
    y = np.array(accs)

    for i in range(len(accs)):
        xticks = []
        for j in range(0, len(list_ids[i])):
            if type(list_ids[i][j]) == tuple:
                xticks.append(inv_map[list_ids[i][j]])
            else:
                xticks.append(inv_map[tuple(list_ids[i][j])])
        xticks_all.append(xticks)

    plt.plot(x, y)
    annot_max(x, y)
    plt.xlabel('Model features')
    plt.ylabel('{} accuracy'.format(figname))
    plt.xticks(np.arange(0, len(accs)), xticks_all, rotation=18, fontsize=5)
    plt.xlabel('Model features', fontsize=12)
    plt.savefig('codi/figures/{}'.format(figname), format='pdf')


def get_feature_dict(params):
    """
    Creates a dictionnary that maps the feature to their position in X
    """
    feature_dict = params['feature_dict']
    for key, value in feature_dict.items():
        if len(value) > 1:
            feature_dict[key] = tuple(range(value[0], value[1]))
    return feature_dict


def annot_max(x, y, ax=None):
    """
    Helper function to mark the maximum in a plot
    """
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = 'x={:.3f}, y={:.3f}'.format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=0.72)
    arrowprops = dict(arrowstyle='->', connectionstyle='angle,angleA=0,angleB=60')
    kw = dict(xycoords='data', textcoords='axes fraction',
              arrowprops=arrowprops, bbox=bbox_props, ha='right', va='top')
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)


def one_hot_encoder(y):
    """
    Creates one-hot-encoded version of target vector y for a binary
    classification task.
    """
    y_ohe = np.zeros((np.shape(y)[0], 2))
    y_ohe[y == 0, 0] = 1
    y_ohe[y == 1, 1] = 1
    return y_ohe


def merge_features(features):
    """
    Merges features into one X array
    """
    X = np.empty((np.shape(features[0])[1], 0))
    for feature in features:
        for feat in feature:
            if len(feat.shape) == 1:
                feat = np.expand_dims(feat, axis=1)
            X = np.concatenate([X, feat], axis=1)
    return X

