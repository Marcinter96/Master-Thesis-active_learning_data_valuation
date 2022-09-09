"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import time
import numpy as np
import torch
import pandas as pd
import random
import statistics

from codi.nlp_trainer import NLPTrainer
from dataset.dataset_loader import Dataset
from inference_models.__inference_utils import compute_percent_correct
from inference_models.__inference_utils import epoch_time
from inference_models.inference_torch import InferenceTorch

"""
This script is designed for the N-step retraining setup.
The inference model is iteratively retrained with more and more points with pseudo-labels.
"""


def main():

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # Initial load of the spacy english tokenizer
    torch.backends.cudnn.deterministic = True
    dataset_name, hyper_yaml = 'trec', 'yaml_hyper/trec_hyper.yaml'
    dataset_loading = Dataset(dataset_name, hyper_yaml)
    dataset_loading.dataset_to_torch()

    print('***TREC Dataset loaded***')

    inference_train_iterator, inference_val_iterator, inference_test_iterator = dataset_loading.get_inference_iterator()
    text, _ = dataset_loading.get_text_label()

    vocab_size = len(text.vocab)
    pad_idx = text.vocab.stoi[text.pad_token]
    inference_yaml = 'trec.yaml'

    inference_model = InferenceTorch(dataset_name, hyper_yaml, vocab_size, pad_idx, dataset_loading.get_device())
    inference_model.load_from_yaml(inference_yaml)
    initial_acc, initial_f1 = inference_model.train_model(text, inference_train_iterator, inference_val_iterator,
                                                        inference_test_iterator)
    print("Initial accuraccy :", initial_acc)
    print("Initial f1 :", initial_f1)
    print('***Inference Model trained, now inferring labels.***')

    # Those arrays store the scores for performance metrics
    initial_accuracies = np.array([initial_acc, initial_f1])

    new_dataset = dataset_loading.copy()

    grad_test = inference_model.compute_grad_test(inference_test_iterator)

    full_hessian = inference_model.compute_full_hessian(inference_train_iterator)

    unlabelled_iterator = new_dataset.get_unlabelled_iterator()
    inference_model.infer_labels(unlabelled_iterator, text, grad=False, inf=True, grad_test=grad_test, hessian=full_hessian)#grad_test=grad_test, hessian=full_hessian
    print('Percent correct on unlabelled dataset prediction ',
        compute_percent_correct(new_dataset.get_unlabelled_dataset().examples))

    print('***Unlabelled dataset processed.***')

    new_dataset = dataset_loading.copy()
    # Beginning of retraining experiments
    unlabelled_dataset_original = new_dataset.get_unlabelled_dataset()
    inference_train_original = new_dataset.get_inference_dataset()

    start_step = time.time()

    # The exp1_percentage is the percentage of points taken at each step
    # In this case, it is a fixed value, but could be changed step after step
    inference_train, inference_god_train, unlabelled_dataset, percent_correct, size_indices, index = \
        inference_model.gradient_retraining(inference_train_original, unlabelled_dataset_original,
                                            inf_rule=True, size=800)

    #influence_dataset, influence_iterator = new_dataset.subset_dataset(index)
    #influence_score = inference_model.compute_influence_score(influence_iterator, full_hessian, grad_test)

    print('Percent Correct for this threshold is {}, for {} points'.format(percent_correct, size_indices))
    new_dataset.update_datasets(inference_god_train, unlabelled_dataset)


    inference_train_iterator, inference_val_iterator, inference_test_iterator = new_dataset. \
        get_inference_iterator()
    text, _ = new_dataset.get_text_label()
    vocab_size = len(text.vocab)

    del inference_model
    torch.cuda.empty_cache()
    inference_model = InferenceTorch(dataset_name, hyper_yaml, vocab_size, pad_idx,
                                     new_dataset.get_device())
    inference_model.load_from_yaml(inference_yaml)

    test_acc, test_f1 = inference_model.train_model(text, inference_train_iterator, inference_val_iterator,
                                                    inference_test_iterator)


    end_step = time.time()

    step_mins, step_secs = epoch_time(start_step, end_step)
    print(f'Time for step: {step_mins}m {step_secs}s')

if __name__ == '__main__':
    main()
