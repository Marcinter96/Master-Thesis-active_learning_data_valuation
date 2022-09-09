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
This script is designed for Active learning set up.
The inference model is retrained once with more and more points manually correctly labelled.
"""


def main():

    exp_number = 15
    data_plot_rand = pd.DataFrame()
    data_init = pd.DataFrame()
    data_plot_grad = pd.DataFrame()
    data_inf_func = pd.DataFrame()

    #torch.manual_seed(0)
    #torch.cuda.manual_seed(0)
    #np.random.seed(0)
    #random.seed(0)
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
    xtra_init = {'acc': initial_accuracies}
    data_init = data_init.append(pd.DataFrame(xtra_init))

    new_dataset = dataset_loading.copy()

    #grad_test = inference_model.compute_grad_test(inference_test_iterator)

    #full_hessian = inference_model.compute_full_hessian(inference_train_iterator)

    # inference_model.infer_training(inference_train_iterator, grad_test, full_hessian)

    codi_labelled_iterator = new_dataset.get_codi_iterator()
    inference_model.infer_labels(codi_labelled_iterator, text)#grad = True, inf=True, grad_test=grad_test,hessian=full_hessian

    print('***CoDi labelled dataset processed.***')

    codi_trainer = NLPTrainer(yaml_model_path='codi/mlp_codi.yaml', yaml_train_path='codi/nlp_trainer_n_step.yaml')

    codi_trainer.create_labelled_dataset(codi_labelled_iterator)
    codi_trainer.train()
    print('***CoDi model trained.***')

    unlabelled_iterator = new_dataset.get_unlabelled_iterator()
    inference_model.infer_labels(unlabelled_iterator, text)

    _ = codi_trainer.create_unlabelled_dataset(unlabelled_iterator)

    print('Percent correct on unlabelled dataset prediction ',
        compute_percent_correct(new_dataset.get_unlabelled_dataset().examples))

    print('***Unlabelled dataset processed, now processing CoDi dataset.***')
    print('***Prediction done.***')

    for ints in range(exp_number):
        print("**** RUN NUMBER {} ****".format(ints))

        start_time = time.time()

        for size in [200, 400, 600, 800, 1000]:


            new_dataset = dataset_loading.copy()
            # Beginning of retraining experiments
            unlabelled_dataset_original = new_dataset.get_unlabelled_dataset()
            inference_train_original = new_dataset.get_inference_dataset()

            start_step = time.time()

            # The exp1_percentage is the percentage of points taken at each step
            # In this case, it is a fixed value, but could be changed step after step
            inference_dataset, unlabelled_dataset, percent_correct, size_indices = \
                inference_model.process_retraining(inference_train_original, unlabelled_dataset_original, 0,
                                                   take_random=True, random_size=size)

            print('Percent Correct for this threshold is {}, for {} points'.format(percent_correct, size_indices))
            new_dataset.update_datasets(inference_dataset, unlabelled_dataset)

            inference_train_iterator, inference_val_iterator, inference_test_iterator = new_dataset. \
                get_inference_iterator()
            text, _ = new_dataset.get_text_label()
            vocab_size = len(text.vocab)

            del inference_model
            torch.cuda.empty_cache()
            inference_model = InferenceTorch(dataset_name, hyper_yaml, vocab_size, pad_idx, new_dataset.get_device())
            inference_model.load_from_yaml(inference_yaml)

            test_acc, test_f1 = inference_model.train_model(text, inference_train_iterator, inference_val_iterator,
                                                        inference_test_iterator)

            end_step = time.time()

            step_mins, step_secs = epoch_time(start_step, end_step)
            print(f'Time for step : {step_mins}m {step_secs}s')

            torch.cuda.empty_cache()
            del new_dataset
            dataset_loading.remove_update_datasets(inference_train_original,unlabelled_dataset_original)

            # Save results for random sampling.
            xtra_rand = {'Added points': [size], 'f1': [test_f1], 'accuracy': [test_acc]}
            data_plot_rand = data_plot_rand.append(pd.DataFrame(xtra_rand))

            print("Marginal increase for Accuracy adding random point is :",
                round((test_acc - initial_acc), 2) * 100, "%")
            print("Marginal increase for F1-score adding random point is :",
                round((test_f1 - initial_f1), 2) * 100, "%")

            # Testing Experience 2  Grad Rule
            new_dataset = dataset_loading.copy()
            # Beginning of retraining experiments
            unlabelled_dataset_original = new_dataset.get_unlabelled_dataset()
            inference_train_original = new_dataset.get_inference_dataset()

            start_step = time.time()

            # The exp1_percentage is the percentage of points taken at each step
            # In this case, it is a fixed value, but could be changed step after step
            inference_train, unlabelled_dataset, percent_correct, size_indices = \
                inference_model.process_retraining(inference_train_original, unlabelled_dataset_original, 0,
                                                   exp1_size=size)

            print('Percent Correct for this threshold is {}, for {} points'.format(percent_correct, size_indices))
            new_dataset.update_datasets(inference_train, unlabelled_dataset)

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

            torch.cuda.empty_cache()
            del new_dataset
            dataset_loading.remove_update_datasets(inference_train_original,unlabelled_dataset_original)

            xtra_grad = {'Added points': [size_indices], 'f1': [test_f1], 'accuracy': [test_acc]}
            data_plot_grad = data_plot_grad.append(pd.DataFrame(xtra_grad))

            print("Marginal increase for Accuracy adding most informative point is :",
                round((test_acc - initial_acc),2)*100, "%")
            print("Marginal increase for F1-score adding most informative point is :",
                round((test_f1 - initial_f1),2)*100, "%")

            # Testing Experience 2  Grad Rule
            new_dataset = dataset_loading.copy()
            # Beginning of retraining experiments
            unlabelled_dataset_original = new_dataset.get_unlabelled_dataset()
            inference_train_original = new_dataset.get_inference_dataset()

            start_step = time.time()

            # The exp1_percentage is the percentage of points taken at each step
            # In this case, it is a fixed value, but could be changed step after step
            inference_dataset, unlabelled_dataset, percent_correct, size_indices = \
                inference_model.process_retraining(inference_train_original, unlabelled_dataset_original, 0,
                                                   log_rule=True, exp1_size=size)

            print('Percent Correct for this threshold is {}, for {} points'.format(percent_correct, size_indices))
            new_dataset.update_datasets(inference_dataset, unlabelled_dataset)

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

            torch.cuda.empty_cache()
            del new_dataset
            dataset_loading.remove_update_datasets(inference_train_original, unlabelled_dataset_original)

            xtra_inf = {'Added points': [size_indices], 'f1': [test_f1], 'accuracy': [test_acc]}
            data_inf_func = data_inf_func.append(pd.DataFrame(xtra_inf))

            print("Marginal increase for Accuracy adding most informative inf point is :",
                  round((test_acc - initial_acc), 2) * 100, "%")
            print("Marginal increase for F1-score adding most informative inf point is :",
                  round((test_f1 - initial_f1), 2) * 100, "%")

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Total time: {epoch_mins}m {epoch_secs}s')

    data_plot_rand.to_csv("TREC/Last_Random2.csv")
    data_plot_grad.to_csv("TREC/Last_Algo2.csv")
    data_init.to_csv("TREC/Initial_last2.csv")
    data_inf_func.to_csv("TREC/Inf_algo2.csv")

def main_codi_grad():

    exp_number = 50
    data_plot_rand = pd.DataFrame()
    data_init = pd.DataFrame()
    data_plot_grad = pd.DataFrame()
    data_inf_func = pd.DataFrame()

    #torch.manual_seed(0)
    #torch.cuda.manual_seed(0)
    #np.random.seed(0)
    #random.seed(0)
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
    xtra_init = {'acc': initial_accuracies}
    data_init = data_init.append(pd.DataFrame(xtra_init))

    new_dataset = dataset_loading.copy()

    #grad_test = inference_model.compute_grad_test(inference_test_iterator)

    #full_hessian = inference_model.compute_full_hessian(inference_train_iterator)

    # inference_model.infer_training(inference_train_iterator, grad_test, full_hessian)

    codi_labelled_iterator = new_dataset.get_codi_iterator()
    inference_model.infer_labels(codi_labelled_iterator, text, grad=True)#grad = True, inf=True, grad_test=grad_test,hessian=full_hessian

    print('***CoDi labelled dataset processed.***')

    codi_trainer = NLPTrainer(yaml_model_path='codi/mlp_codi.yaml', yaml_train_path='codi/nlp_experience3.yaml')

    codi_trainer.create_labelled_dataset(codi_labelled_iterator)
    codi_trainer.train()
    print('***CoDi model trained.***')

    unlabelled_iterator = new_dataset.get_unlabelled_iterator()
    inference_model.infer_labels(unlabelled_iterator, text, grad=True)

    _ = codi_trainer.create_unlabelled_dataset(unlabelled_iterator)

    print('Percent correct on unlabelled dataset prediction ',
        compute_percent_correct(new_dataset.get_unlabelled_dataset().examples))

    print('***Unlabelled dataset processed, now processing CoDi dataset.***')
    print('***Prediction done.***')

    for ints in range(exp_number):
        print("**** RUN NUMBER {} ****".format(ints))

        start_time = time.time()

        for size in [200, 400, 600, 800, 1000]:


            new_dataset = dataset_loading.copy()
            # Beginning of retraining experiments
            unlabelled_dataset_original = new_dataset.get_unlabelled_dataset()
            inference_train_original = new_dataset.get_inference_dataset()

            start_step = time.time()

            # The exp1_percentage is the percentage of points taken at each step
            # In this case, it is a fixed value, but could be changed step after step
            inference_dataset, unlabelled_dataset, percent_correct, size_indices = \
                inference_model.process_retraining(inference_train_original, unlabelled_dataset_original, 0,
                                                   take_random=True, random_size=size)

            print('Percent Correct for this threshold is {}, for {} points'.format(percent_correct, size_indices))
            new_dataset.update_datasets(inference_dataset, unlabelled_dataset)

            inference_train_iterator, inference_val_iterator, inference_test_iterator = new_dataset. \
                get_inference_iterator()
            text, _ = new_dataset.get_text_label()
            vocab_size = len(text.vocab)

            del inference_model
            torch.cuda.empty_cache()
            inference_model = InferenceTorch(dataset_name, hyper_yaml, vocab_size, pad_idx, new_dataset.get_device())
            inference_model.load_from_yaml(inference_yaml)

            test_acc, test_f1 = inference_model.train_model(text, inference_train_iterator, inference_val_iterator,
                                                        inference_test_iterator)

            end_step = time.time()

            step_mins, step_secs = epoch_time(start_step, end_step)
            print(f'Time for step : {step_mins}m {step_secs}s')

            torch.cuda.empty_cache()
            del new_dataset
            dataset_loading.remove_update_datasets(inference_train_original,unlabelled_dataset_original)

            # Save results for random sampling.
            xtra_rand = {'Added points': [size], 'f1': [test_f1], 'accuracy': [test_acc]}
            data_plot_rand = data_plot_rand.append(pd.DataFrame(xtra_rand))

            print("Marginal increase for Accuracy adding random point is :",
                round((test_acc - initial_acc), 2) * 100, "%")
            print("Marginal increase for F1-score adding random point is :",
                round((test_f1 - initial_f1), 2) * 100, "%")

            # Testing Experience 2  Grad Rule
            new_dataset = dataset_loading.copy()
            # Beginning of retraining experiments
            unlabelled_dataset_original = new_dataset.get_unlabelled_dataset()
            inference_train_original = new_dataset.get_inference_dataset()

            start_step = time.time()

            # The exp1_percentage is the percentage of points taken at each step
            # In this case, it is a fixed value, but could be changed step after step
            inference_train, unlabelled_dataset, percent_correct, size_indices = \
                inference_model.process_retraining(inference_train_original, unlabelled_dataset_original, 0,
                                                   exp1_size=size)

            print('Percent Correct for this threshold is {}, for {} points'.format(percent_correct, size_indices))
            new_dataset.update_datasets(inference_train, unlabelled_dataset)

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

            torch.cuda.empty_cache()
            del new_dataset
            dataset_loading.remove_update_datasets(inference_train_original,unlabelled_dataset_original)

            xtra_grad = {'Added points': [size_indices], 'f1': [test_f1], 'accuracy': [test_acc]}
            data_plot_grad = data_plot_grad.append(pd.DataFrame(xtra_grad))

            print("Marginal increase for Accuracy adding most informative point is :",
                round((test_acc - initial_acc),2)*100, "%")
            print("Marginal increase for F1-score adding most informative point is :",
                round((test_f1 - initial_f1),2)*100, "%")

        #     # Testing Experience 2  Grad Rule
        #     new_dataset = dataset_loading.copy()
        #     # Beginning of retraining experiments
        #     unlabelled_dataset_original = new_dataset.get_unlabelled_dataset()
        #     inference_train_original = new_dataset.get_inference_dataset()
        #
        #     start_step = time.time()
        #
        #     # The exp1_percentage is the percentage of points taken at each step
        #     # In this case, it is a fixed value, but could be changed step after step
        #     inference_dataset, unlabelled_dataset, percent_correct, size_indices = \
        #         inference_model.process_retraining(inference_train_original, unlabelled_dataset_original, 0,
        #                                            scoring_rule=True, exp1_size=size)
        #
        #     print('Percent Correct for this threshold is {}, for {} points'.format(percent_correct, size_indices))
        #     new_dataset.update_datasets(inference_dataset, unlabelled_dataset)
        #
        #     inference_train_iterator, inference_val_iterator, inference_test_iterator = new_dataset. \
        #         get_inference_iterator()
        #     text, _ = new_dataset.get_text_label()
        #     vocab_size = len(text.vocab)
        #
        #     del inference_model
        #     torch.cuda.empty_cache()
        #     inference_model = InferenceTorch(dataset_name, hyper_yaml, vocab_size, pad_idx,
        #                                      new_dataset.get_device())
        #     inference_model.load_from_yaml(inference_yaml)
        #
        #     test_acc, test_f1 = inference_model.train_model(text, inference_train_iterator, inference_val_iterator,
        #                                                     inference_test_iterator)
        #
        #     end_step = time.time()
        #
        #     step_mins, step_secs = epoch_time(start_step, end_step)
        #     print(f'Time for step: {step_mins}m {step_secs}s')
        #
        #     torch.cuda.empty_cache()
        #     del new_dataset
        #     dataset_loading.remove_update_datasets(inference_train_original, unlabelled_dataset_original)
        #
        #     xtra_inf = {'Added points': [size_indices], 'f1': [test_f1], 'accuracy': [test_acc]}
        #     data_inf_func = data_inf_func.append(pd.DataFrame(xtra_inf))
        #
        #     print("Marginal increase for Accuracy adding most informative inf point is :",
        #           round((test_acc - initial_acc), 2) * 100, "%")
        #     print("Marginal increase for F1-score adding most informative inf point is :",
        #           round((test_f1 - initial_f1), 2) * 100, "%")
        #
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Total time: {epoch_mins}m {epoch_secs}s')

    data_plot_rand.to_csv("TREC/Last_Random_codi_grad.csv")
    data_plot_grad.to_csv("TREC/Last_Algo_codi_grad.csv")
    data_init.to_csv("TREC/Initial_last_codi_grad.csv")
    data_inf_func.to_csv("TREC/Inf_algo_codi_grad.csv")

def main_inf():
    exp_number = 50
    data_plot_rand = pd.DataFrame()
    data_init = pd.DataFrame()
    data_plot_grad = pd.DataFrame()
    data_inf_func = pd.DataFrame()

    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # np.random.seed(0)
    # random.seed(0)
    # Initial load of the spacy english tokenizer
    torch.backends.cudnn.deterministic = True
    dataset_name, hyper_yaml = 'imdb', 'yaml_hyper/imdb_hyper.yaml'
    dataset_loading = Dataset(dataset_name, hyper_yaml)
    dataset_loading.dataset_to_torch()

    print('***TREC Dataset loaded***')

    inference_train_iterator, inference_val_iterator, inference_test_iterator = dataset_loading.get_inference_iterator()
    text, _ = dataset_loading.get_text_label()

    vocab_size = len(text.vocab)
    pad_idx = text.vocab.stoi[text.pad_token]
    inference_yaml = 'imdb.yaml'

    inference_model = InferenceTorch(dataset_name, hyper_yaml, vocab_size, pad_idx, dataset_loading.get_device())
    inference_model.load_from_yaml(inference_yaml)
    initial_acc, initial_f1 = inference_model.train_model(text, inference_train_iterator, inference_val_iterator,
                                                          inference_test_iterator)
    print("Initial accuraccy :", initial_acc)
    print("Initial f1 :", initial_f1)
    print('***Inference Model trained, now inferring labels.***')

    # Those arrays store the scores for performance metrics
    initial_accuracies = np.array([initial_acc, initial_f1])
    xtra_init = {'acc': initial_accuracies}
    data_init = data_init.append(pd.DataFrame(xtra_init))

    new_dataset = dataset_loading.copy()

    grad_test = inference_model.compute_grad_test(inference_test_iterator)

    full_hessian = inference_model.compute_full_hessian(inference_train_iterator)

    inference_model.infer_training(inference_train_iterator, grad_test, full_hessian)

    codi_labelled_iterator = new_dataset.get_codi_iterator()
    inference_model.infer_labels(codi_labelled_iterator, text,
                                 grad=True, inf=True, grad_test=grad_test,
                                 hessian=full_hessian)  # grad = True, inf=True, grad_test=grad_test,hessian=full_hessian

    print('***CoDi labelled dataset processed.***')

    codi_trainer = NLPTrainer(yaml_model_path='codi/mlp_codi.yaml', yaml_train_path='codi/nlp_experience4.yaml')

    codi_trainer.create_labelled_dataset(codi_labelled_iterator)
    codi_trainer.train()
    print('***CoDi model trained.***')

    unlabelled_iterator = new_dataset.get_unlabelled_iterator()
    inference_model.infer_labels(unlabelled_iterator, text,
                                 grad=True, inf=True, grad_test=grad_test, hessian=full_hessian)

    _ = codi_trainer.create_unlabelled_dataset(unlabelled_iterator)

    print('Percent correct on unlabelled dataset prediction ',
          compute_percent_correct(new_dataset.get_unlabelled_dataset().examples))

    print('***Unlabelled dataset processed, now processing CoDi dataset.***')
    print('***Prediction done.***')

    for ints in range(exp_number):
        print("**** RUN NUMBER {} ****".format(ints))

        start_time = time.time()

        for size in [200, 400, 600, 800, 1000]:
            new_dataset = dataset_loading.copy()
            # Beginning of retraining experiments
            unlabelled_dataset_original = new_dataset.get_unlabelled_dataset()
            inference_train_original = new_dataset.get_inference_dataset()

            start_step = time.time()

            # The exp1_percentage is the percentage of points taken at each step
            # In this case, it is a fixed value, but could be changed step after step
            inference_dataset, unlabelled_dataset, percent_correct, size_indices = \
                inference_model.process_retraining(inference_train_original, unlabelled_dataset_original, 0,
                                                   take_random=True, random_size=size)

            print('Percent Correct for this threshold is {}, for {} points'.format(percent_correct, size_indices))
            new_dataset.update_datasets(inference_dataset, unlabelled_dataset)

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
            print(f'Time for step : {step_mins}m {step_secs}s')

            torch.cuda.empty_cache()
            del new_dataset
            dataset_loading.remove_update_datasets(inference_train_original, unlabelled_dataset_original)

            # Save results for random sampling.
            xtra_rand = {'Added points': [size], 'f1': [test_f1], 'accuracy': [test_acc]}
            data_plot_rand = data_plot_rand.append(pd.DataFrame(xtra_rand))

            print("Marginal increase for Accuracy adding random point is :",
                  round((test_acc - initial_acc), 2) * 100, "%")
            print("Marginal increase for F1-score adding random point is :",
                  round((test_f1 - initial_f1), 2) * 100, "%")

            # Testing Experience 2  Grad Rule
            new_dataset = dataset_loading.copy()
            # Beginning of retraining experiments
            unlabelled_dataset_original = new_dataset.get_unlabelled_dataset()
            inference_train_original = new_dataset.get_inference_dataset()

            start_step = time.time()

            # The exp1_percentage is the percentage of points taken at each step
            # In this case, it is a fixed value, but could be changed step after step
            inference_train, unlabelled_dataset, percent_correct, size_indices = \
                inference_model.process_retraining(inference_train_original, unlabelled_dataset_original, 0,
                                                   exp1_size=size)

            print('Percent Correct for this threshold is {}, for {} points'.format(percent_correct, size_indices))
            new_dataset.update_datasets(inference_train, unlabelled_dataset)

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

            torch.cuda.empty_cache()
            del new_dataset
            dataset_loading.remove_update_datasets(inference_train_original, unlabelled_dataset_original)

            xtra_grad = {'Added points': [size_indices], 'f1': [test_f1], 'accuracy': [test_acc]}
            data_plot_grad = data_plot_grad.append(pd.DataFrame(xtra_grad))

            print("Marginal increase for Accuracy adding most informative point is :",
                  round((test_acc - initial_acc), 2) * 100, "%")
            print("Marginal increase for F1-score adding most informative point is :",
                  round((test_f1 - initial_f1), 2) * 100, "%")

            # # Testing Experience 2  Grad Rule
            # new_dataset = dataset_loading.copy()
            # # Beginning of retraining experiments
            # unlabelled_dataset_original = new_dataset.get_unlabelled_dataset()
            # inference_train_original = new_dataset.get_inference_dataset()
            #
            # start_step = time.time()
            #
            # # The exp1_percentage is the percentage of points taken at each step
            # # In this case, it is a fixed value, but could be changed step after step
            # inference_dataset, unlabelled_dataset, percent_correct, size_indices = \
            #     inference_model.process_retraining(inference_train_original, unlabelled_dataset_original, 0,
            #                                        scoring_rule=True, exp1_size=size)
            #
            # print('Percent Correct for this threshold is {}, for {} points'.format(percent_correct, size_indices))
            # new_dataset.update_datasets(inference_dataset, unlabelled_dataset)
            #
            # inference_train_iterator, inference_val_iterator, inference_test_iterator = new_dataset. \
            #     get_inference_iterator()
            # text, _ = new_dataset.get_text_label()
            # vocab_size = len(text.vocab)
            #
            # del inference_model
            # torch.cuda.empty_cache()
            # inference_model = InferenceTorch(dataset_name, hyper_yaml, vocab_size, pad_idx,
            #                                  new_dataset.get_device())
            # inference_model.load_from_yaml(inference_yaml)
            #
            # test_acc, test_f1 = inference_model.train_model(text, inference_train_iterator, inference_val_iterator,
            #                                                 inference_test_iterator)
            #
            # end_step = time.time()
            #
            # step_mins, step_secs = epoch_time(start_step, end_step)
            # print(f'Time for step: {step_mins}m {step_secs}s')
            #
            # torch.cuda.empty_cache()
            # del new_dataset
            # dataset_loading.remove_update_datasets(inference_train_original, unlabelled_dataset_original)
            #
            # xtra_inf = {'Added points': [size_indices], 'f1': [test_f1], 'accuracy': [test_acc]}
            # data_inf_func = data_inf_func.append(pd.DataFrame(xtra_inf))
            #
            # print("Marginal increase for Accuracy adding most informative inf point is :",
            #       round((test_acc - initial_acc), 2) * 100, "%")
            # print("Marginal increase for F1-score adding most informative inf point is :",
            #       round((test_f1 - initial_f1), 2) * 100, "%")

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Total time: {epoch_mins}m {epoch_secs}s')

    data_plot_rand.to_csv("TREC/Last_Random_inf.csv")
    data_plot_grad.to_csv("TREC/Last_Algo_inf.csv")
    data_init.to_csv("TREC/Initial_last_inf.csv")
    data_inf_func.to_csv("TREC/Inf_algo_inf.csv")


def main_grad():

    exp_number = 10
    data_plot_rand = pd.DataFrame()
    data_init = pd.DataFrame()
    data_plot_grad = pd.DataFrame()

    #torch.manual_seed(0)
    #torch.cuda.manual_seed(0)
    #np.random.seed(0)
    #random.seed(0)
    # Initial load of the spacy english tokenizer
    torch.backends.cudnn.deterministic = True
    dataset_name, hyper_yaml = 'imdb', 'yaml_hyper/imdb_hyper1.yaml'
    dataset_loading = Dataset(dataset_name, hyper_yaml)
    dataset_loading.dataset_to_torch()

    print('***TREC Dataset loaded***')

    inference_train_iterator, inference_val_iterator, inference_test_iterator = dataset_loading.get_inference_iterator()
    text, _ = dataset_loading.get_text_label()

    vocab_size = len(text.vocab)
    pad_idx = text.vocab.stoi[text.pad_token]
    inference_yaml = 'imdb.yaml'
    hyper_yaml = 'yaml_hyper/imdb_hyper.yaml'
    inference_model = InferenceTorch(dataset_name, hyper_yaml, vocab_size, pad_idx, dataset_loading.get_device())
    inference_model.load_from_yaml(inference_yaml)
    initial_acc, initial_f1 = inference_model.train_model(text, inference_train_iterator, inference_val_iterator,
                                                        inference_test_iterator)
    print("Initial accuraccy :", initial_acc)
    print("Initial f1 :", initial_f1)
    print('***Inference Model trained, now inferring labels.***')


    # Those arrays store the scores for performance metrics
    initial_accuracies = np.array([initial_acc, initial_f1])
    xtra_init = {'acc': initial_accuracies}
    data_init = data_init.append(pd.DataFrame(xtra_init))

    new_dataset = dataset_loading.copy()

    #grad_test = inference_model.compute_grad_test(inference_test_iterator)

    #full_hessian = inference_model.compute_full_hessian(inference_train_iterator)

    # inference_model.infer_training(inference_train_iterator, grad_test, full_hessian)

    # codi_labelled_iterator = new_dataset.get_codi_iterator()
    # inference_model.infer_labels(codi_labelled_iterator, text, grad=True)#grad = True, inf=True, grad_test=grad_test,hessian=full_hessian
    #
    # print('***CoDi labelled dataset processed.***')
    #
    # codi_trainer = NLPTrainer(yaml_model_path='codi/mlp_codi.yaml', yaml_train_path='codi/nlp_trainer_n_step.yaml')
    #
    # codi_trainer.create_labelled_dataset(codi_labelled_iterator)
    # codi_trainer.train()
    # print('***CoDi model trained.***')

    unlabelled_iterator = new_dataset.get_unlabelled_iterator()
    inference_model.infer_labels(unlabelled_iterator, text, grad=True)

    #_ = codi_trainer.create_unlabelled_dataset(unlabelled_iterator)

    print('Percent correct on unlabelled dataset prediction ',
        compute_percent_correct(new_dataset.get_unlabelled_dataset().examples))

    print('***Unlabelled dataset processed, now processing CoDi dataset.***')
    print('***Prediction done.***')

    for ints in range(exp_number):
        print("**** RUN NUMBER {} ****".format(ints))

        start_time = time.time()

        for size in [200, 400, 600, 800, 1000]:


            new_dataset = dataset_loading.copy()
            # Beginning of retraining experiments
            unlabelled_dataset_original = new_dataset.get_unlabelled_dataset()
            inference_train_original = new_dataset.get_inference_dataset()

            start_step = time.time()

            # The exp1_percentage is the percentage of points taken at each step
            # In this case, it is a fixed value, but could be changed step after step
            inference_dataset, unlabelled_dataset, percent_correct, size_indices = \
                inference_model.gradient_retraining(inference_train_original, unlabelled_dataset_original,
                                                   take_random=True, random_size=size)

            print('Percent Correct for this threshold is {}, for {} points'.format(percent_correct, size_indices))
            new_dataset.update_datasets(inference_dataset, unlabelled_dataset)

            inference_train_iterator, inference_val_iterator, inference_test_iterator = new_dataset. \
                get_inference_iterator()
            text, _ = new_dataset.get_text_label()
            vocab_size = len(text.vocab)

            del inference_model
            torch.cuda.empty_cache()
            inference_model = InferenceTorch(dataset_name, hyper_yaml, vocab_size, pad_idx, new_dataset.get_device())
            inference_model.load_from_yaml(inference_yaml)

            test_acc, test_f1 = inference_model.train_model(text, inference_train_iterator, inference_val_iterator,
                                                        inference_test_iterator)

            end_step = time.time()

            step_mins, step_secs = epoch_time(start_step, end_step)
            print(f'Time for step : {step_mins}m {step_secs}s')

            torch.cuda.empty_cache()
            del new_dataset
            dataset_loading.remove_update_datasets(inference_train_original,unlabelled_dataset_original)

            # Save results for random sampling.
            xtra_rand = {'Added points': [size], 'f1': [test_f1], 'accuracy': [test_acc]}
            data_plot_rand = data_plot_rand.append(pd.DataFrame(xtra_rand))

            print("Marginal increase for Accuracy adding random point is :",
                round((test_acc - initial_acc), 2) * 100, "%")
            print("Marginal increase for F1-score adding random point is :",
                round((test_f1 - initial_f1), 2) * 100, "%")

            # Testing Experience 2  Grad Rule
            new_dataset = dataset_loading.copy()
            # Beginning of retraining experiments
            unlabelled_dataset_original = new_dataset.get_unlabelled_dataset()
            inference_train_original = new_dataset.get_inference_dataset()

            start_step = time.time()

            # The exp1_percentage is the percentage of points taken at each step
            # In this case, it is a fixed value, but could be changed step after step
            inference_train, unlabelled_dataset, percent_correct, size_indices = \
                inference_model.gradient_retraining(inference_train_original, unlabelled_dataset_original,
                                                   grad_rule=True, size=size)

            print('Percent Correct for this threshold is {}, for {} points'.format(percent_correct, size_indices))
            new_dataset.update_datasets(inference_train, unlabelled_dataset)

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

            torch.cuda.empty_cache()
            del new_dataset
            dataset_loading.remove_update_datasets(inference_train_original,unlabelled_dataset_original)

            xtra_grad = {'Added points': [size_indices], 'f1': [test_f1], 'accuracy': [test_acc]}
            data_plot_grad = data_plot_grad.append(pd.DataFrame(xtra_grad))

            print("Marginal increase for Accuracy adding most informative point is :",
                round((test_acc - initial_acc),2)*100, "%")
            print("Marginal increase for F1-score adding most informative point is :",
                round((test_f1 - initial_f1),2)*100, "%")

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Total time: {epoch_mins}m {epoch_secs}s')

    data_plot_rand.to_csv("TREC/Last_Random_grad.csv")
    data_plot_grad.to_csv("TREC/Last_Algo_grad.csv")
    data_init.to_csv("TREC/Initial_last_grad.csv")


def main_neigb():

    exp_number = 7
    data_plot_rand = pd.DataFrame()
    data_init = pd.DataFrame()
    data_plot_grad = pd.DataFrame()

    #torch.manual_seed(0)
    #torch.cuda.manual_seed(0)
    #np.random.seed(0)
    #random.seed(0)
    # Initial load of the spacy english tokenizer
    torch.backends.cudnn.deterministic = True
    dataset_name, hyper_yaml = 'trec', 'yaml_hyper/trec_hyper2.yaml'
    dataset_loading = Dataset(dataset_name, hyper_yaml)
    dataset_loading.dataset_to_torch()

    print('***TREC Dataset loaded***')

    inference_train_iterator, inference_val_iterator, inference_test_iterator = dataset_loading.get_inference_iterator()
    text, _ = dataset_loading.get_text_label()

    vocab_size = len(text.vocab)
    pad_idx = text.vocab.stoi[text.pad_token]
    inference_yaml = 'trec.yaml'

    hyper_yaml = 'yaml_hyper/trec_hyper.yaml'
    inference_model = InferenceTorch(dataset_name, hyper_yaml, vocab_size, pad_idx, dataset_loading.get_device())
    inference_model.load_from_yaml(inference_yaml)
    initial_acc, initial_f1 = inference_model.train_model(text, inference_train_iterator, inference_val_iterator,
                                                        inference_test_iterator)
    print("Initial accuraccy :", initial_acc)
    print("Initial f1 :", initial_f1)
    print('***Inference Model trained, now inferring labels.***')


    # Those arrays store the scores for performance metrics
    initial_accuracies = np.array([initial_acc, initial_f1])
    xtra_init = {'acc': initial_accuracies}
    data_init = data_init.append(pd.DataFrame(xtra_init))

    new_dataset = dataset_loading.copy()

    grad_test = inference_model.compute_grad_test(inference_test_iterator)

    full_hessian = inference_model.compute_full_hessian(inference_train_iterator)

    inference_model.infer_training(inference_train_iterator, grad_test, full_hessian)

    # codi_labelled_iterator = new_dataset.get_codi_iterator()
    # inference_model.infer_labels(codi_labelled_iterator, text, grad=True)#grad = True, inf=True, grad_test=grad_test,hessian=full_hessian
    #
    # print('***CoDi labelled dataset processed.***')
    #
    # codi_trainer = NLPTrainer(yaml_model_path='codi/mlp_codi.yaml', yaml_train_path='codi/nlp_trainer_n_step.yaml')
    #
    # codi_trainer.create_labelled_dataset(codi_labelled_iterator)
    # codi_trainer.train()
    # print('***CoDi model trained.***')

    unlabelled_iterator = new_dataset.get_unlabelled_iterator()
    inference_model.infer_labels(unlabelled_iterator, text)

    #_ = codi_trainer.create_unlabelled_dataset(unlabelled_iterator)

    print('Percent correct on unlabelled dataset prediction ',
        compute_percent_correct(new_dataset.get_unlabelled_dataset().examples))

    print('***Unlabelled dataset processed, now processing CoDi dataset.***')
    print('***Prediction done.***')

    for ints in range(exp_number):
        print("**** RUN NUMBER {} ****".format(ints))

        start_time = time.time()

        for size in [400, 800, 1200, 1600, 2000]:

            # Testing Experience 2  Grad Rule
            new_dataset = dataset_loading.copy()
            # Beginning of retraining experiments
            unlabelled_dataset_original = new_dataset.get_unlabelled_dataset()
            inference_train_original = new_dataset.get_inference_dataset()

            start_step = time.time()

            # The exp1_percentage is the percentage of points taken at each step
            # In this case, it is a fixed value, but could be changed step after step
            inference_train, unlabelled_dataset, size_indices = \
                inference_model.process_removing(inference_train_original, unlabelled_dataset_original,
                                                 take_random=True, random_size=size)

            new_dataset.update_datasets(inference_train, unlabelled_dataset)

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

            torch.cuda.empty_cache()
            del new_dataset
            dataset_loading.remove_update_datasets(inference_train_original, unlabelled_dataset_original)

            xtra_rand = {'Added points': [size_indices], 'f1': [test_f1], 'accuracy': [test_acc]}
            data_plot_rand = data_plot_rand.append(pd.DataFrame(xtra_rand))

            print("Marginal increase for Accuracy adding random point is :",
                  round((test_acc - initial_acc), 2) * 100, "%")
            print("Marginal increase for F1-score adding random point is :",
                  round((test_f1 - initial_f1), 2) * 100, "%")


            # Testing Experience 2  Grad Rule
            new_dataset = dataset_loading.copy()
            # Beginning of retraining experiments
            unlabelled_dataset_original = new_dataset.get_unlabelled_dataset()
            inference_train_original = new_dataset.get_inference_dataset()

            start_step = time.time()

            # The exp1_percentage is the percentage of points taken at each step
            # In this case, it is a fixed value, but could be changed step after step
            inference_train, unlabelled_dataset, size_indices = \
                inference_model.process_removing(inference_train_original, unlabelled_dataset_original,
                                                   inf_max=True, size=size)

            new_dataset.update_datasets(inference_train, unlabelled_dataset)

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

            torch.cuda.empty_cache()
            del new_dataset
            dataset_loading.remove_update_datasets(inference_train_original,unlabelled_dataset_original)

            xtra_grad = {'Added points': [size_indices], 'f1': [test_f1], 'accuracy': [test_acc]}
            data_plot_grad = data_plot_grad.append(pd.DataFrame(xtra_grad))

            print("Marginal increase for Accuracy adding most informative point is :",
                round((test_acc - initial_acc),2)*100, "%")
            print("Marginal increase for F1-score adding most informative point is :",
                round((test_f1 - initial_f1),2)*100, "%")
            
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Total time: {epoch_mins}m {epoch_secs}s')

    data_plot_rand.to_csv("TREC/Last_Random_neigb.csv")
    data_plot_grad.to_csv("TREC/Last_Algo_neigb.csv")
    data_init.to_csv("TREC/Initial_last_neigb.csv")

if __name__ == '__main__':
    main()