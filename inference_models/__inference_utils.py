"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

import torch
import numpy as np
from features.features_classes.logits_feature import LogitsFeature
from features.features_classes.gradients_features import GradientFeature
from features.features_classes.influence_features import InfluenceFeature

import copy
from scipy.stats import entropy
from sklearn.metrics import f1_score
from torch.autograd import Variable, grad


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch
    :param preds: prediction logits
    :param y: target labels
    :return: accuracy = percentage of correct predictions
    """

    # round predictions to the closest integer
    rounded_predictions = torch.round(torch.sigmoid(preds))
    correct = (rounded_predictions == y).float()
    acc = correct.sum() / len(correct)

    return acc


def binary_f1_score(preds, y):
    """
    Returns F1-score per batch
    :param preds: prediction logits
    :param y: target labels
    :return: score = F1-score
    """
    rounded_predictions = torch.round(torch.sigmoid(preds))

    return f1_score(y.cpu().detach().numpy(), rounded_predictions.cpu().detach().numpy(), average='weighted')


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    :param preds: prediction logits
    :param y: target labels
    :return: categorical accuracy
    """
    max_preds = preds.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(y)

    return correct.sum() / torch.FloatTensor([y.shape[0]])


def categorical_f1_score(preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True).squeeze(1)

    return f1_score(y.cpu().detach().numpy(), max_preds.cpu().detach().numpy(), average='weighted')


def train(model, iterator, optimizer, criterion, binary=True):
    """
    Train a PyTorch model
    :param model: the PyTorch model
    :param iterator: dataset in batch iterator form
    :param optimizer: optimizer for the training
    :param criterion: criterion between predictions and target
    :param binary: whether we work with binary classes or multi-classes
    :return: mean loss and accuracy
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        if binary:
            predictions = model(batch.text).squeeze(1)
        else:
            predictions = model(batch.text)

        loss = criterion(predictions, batch.label)

        if binary:
            acc = binary_accuracy(predictions, batch.label)
        else:
            acc = categorical_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, binary=True):
    """
    Evaluate a PyTorch model given a dataset (in batch iterator form) and a criterion
    :param model: the PyTorch model
    :param iterator: iterator over batches of data
    :param criterion: criterion to be used to compare target and predictions
    :param binary: whether we work with binary classes or multi-classes
    :return: mean loss and accuracy over the provided dataset
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    # Put the model in eval mode
    model.eval()

#    with torch.no_grad():
    for batch in iterator:
        if binary:
            predictions = model(batch.text).squeeze(1)
        else:
            predictions = model(batch.text)

        loss = criterion(predictions, batch.label)

        if binary:
            acc = binary_accuracy(predictions, batch.label)
            f1 = binary_f1_score(predictions, batch.label)
        else:
            acc = categorical_accuracy(predictions, batch.label)
            f1 = categorical_f1_score(predictions, batch.label)

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_f1 += f1.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)


def predict(model, iterator, criterion, optimizer, text, grad, inf, grad_test, hessian, binary=True):
    """
    Predicts logits, gradients influence scores and labels with a trained model and outputs the original labels as well.
    All operations are done in-place on the dataset itself.
    :param model: the PyTorch model
    :param iterator: iterator over batches of data
    :param binary: whether we work with binary classes or multi-classes
    :param criterion: Pytorch criterion
    :param optimizer: Model optimizer
    :param text: Vocabulary of the chosen dataset
    :param grad: if gradient computation
    :param inf: if influence score computation
    :param grad_test: Matrix of gradients derivative for test set
    :param hessian: inverse Hessian matrix
    """
    # Put the model in eval mode
    model.eval()

    #with torch.no_grad():
    for example_batch, batch in iterator:

        if binary:

            predictions_torch = model(batch.text).squeeze(1)
            labels_torch = torch.round(torch.sigmoid(predictions_torch))
            pooling = model.pooling(batch.text)

            if grad:
                original_input_embedding_pred, input_grad_pred = get_pred_gradients(example_batch, predictions_torch,
                                                                                    model, criterion, optimizer,
                                                                                    labels_torch, text)
                original_input_embedding_all, input_grad_all = get_all_gradients(example_batch, predictions_torch,
                                                                                 model, criterion, optimizer, text)

        else:

            predictions_torch = model(batch.text)
            labels_torch = predictions_torch.argmax(dim=1, keepdim=True).squeeze(1)
            pooling = model.pooling(batch.text)

            if grad:

                original_input_embedding_pred, input_grad_pred = get_pred_gradients(example_batch, predictions_torch,
                                                                                    model, criterion, optimizer,
                                                                                    labels_torch, text)
                original_input_embedding_all, input_grad_all = get_all_gradients(example_batch, predictions_torch,
                                                                                 model, criterion, optimizer, text)

        for index, example in enumerate(example_batch):
            # Retrieve the corresponding label
            example.predicted_label = iterator.dataset.fields['label'].vocab.itos[int(labels_torch[index])]

            if example.predicted_label == example.label:
                example.is_correct = True
            else:
                example.is_correct = False

            if binary:

                example.logit = double_logits(predictions_torch[index].sigmoid().detach().cpu().numpy())
                # Get the pooling layer values for visualizing embedding
                example.pooled = pooling[index].detach().cpu().numpy()

                if grad:
                    # Get gradients for each class and also for the predicted class
                    input_grad_index = []
                    original_input_embedding_index = []
                    for i in range(len(input_grad_all)):
                        input_grad_index.append(input_grad_all[i][index])
                        original_input_embedding_index.append(original_input_embedding_all[i][index])
                    grad_scores = mean_grad(original_input_embedding_index, input_grad_index)
                    example.norm_scores =grad_scores.detach().cpu().numpy()
                    pred_grad = mean_grad_pred(original_input_embedding_pred[index],input_grad_pred[index])
                    example.pred_grad = pred_grad.detach().cpu().numpy()

                if inf:
                    # Get influence score for each class and also for the predicted class
                    grad_unlabelled_pred = get_inf_grad(predictions_torch[index], labels_torch[index], model, criterion)
                    all_unlabelled_grad = get_inf_all_grad(predictions_torch[index], model, criterion)
                    inf_unlabelled_pred = get_inf_score(grad_test, hessian, grad_unlabelled_pred)
                    inf_unlabelled_all = get_inf_all_score(grad_test, hessian, all_unlabelled_grad)

                    example.influence_score_pred = inf_unlabelled_pred.detach().cpu().numpy()
                    example.influence_score_all = inf_unlabelled_all.detach().cpu().numpy()

            else:
                temp_logit = predictions_torch[index].detach().cpu().numpy()
                # We avoid negative values for entropy computation
                example.logit = (temp_logit + np.abs(temp_logit))/2
                # Get pooling layer values for neighbor visualization
                example.pooled = pooling[index].detach().cpu().numpy()

                if grad:
                    # Get gradients for each class and also for the predicted class
                    input_grad_index = []
                    original_input_embedding_index = []
                    for i in range(len(input_grad_all)):
                        input_grad_index.append(input_grad_all[i][index])
                        original_input_embedding_index.append(original_input_embedding_all[i][index])
                    grad_scores = mean_grad(original_input_embedding_index, input_grad_index)
                    example.norm_scores =grad_scores.detach().cpu().numpy()
                    pred_grad = mean_grad_pred(original_input_embedding_pred[index],input_grad_pred[index])
                    example.pred_grad = pred_grad.detach().cpu().numpy()

                if inf:
                    # Get influence score for each class and also for the predicted class
                    grad_unlabelled_pred = get_inf_grad(predictions_torch[index], labels_torch[index], model, criterion)
                    all_unlabelled_grad = get_inf_all_grad(predictions_torch[index], model, criterion)
                    inf_unlabelled_pred = get_inf_score(grad_test, hessian, grad_unlabelled_pred)
                    inf_unlabelled_all = get_inf_all_score(grad_test, hessian, all_unlabelled_grad)

                    example.influence_score_pred = inf_unlabelled_pred.detach().cpu().numpy()
                    example.influence_score_all = inf_unlabelled_all.detach().cpu().numpy()

            logits_features = LogitsFeature()
            logits_features.augment(np.expand_dims(example.logit, axis=0))
            _, example.margin, example.ratio, example.entropy = logits_features.get_features()
            if grad:

                gradient_features = GradientFeature(model.fc.out_features)
                gradient_features.augment(np.expand_dims(example.norm_scores, axis=0))
                _, example.grad_variance, example.grad_min_margin, example.grad_min = gradient_features.get_features()
            if inf:

                influence_features = InfluenceFeature(model.fc.out_features)
                influence_features.augment(np.expand_dims(example.influence_score_all, axis=0))
                _, example.influence_variance, example.influence_min_margin, example.influence_min = influence_features.get_features()


def get_all_gradients(original_sentence, prediction, model, criterion, optimizer, TEXT):
    """
    Get all gradients for each class (in batch iterator form) and a criterion
    :param model: the PyTorch model
    :param original_sentence: length of the data in the batch
    :param criterion: criterion to be used to compare target and predictions
    :param prediction: models prediction for the batch
    :param optimizer: Optimizer for the model
    :param TEXT: Dataset vocabulary
    :return: list containing gradients update with respect of the input for each class
    :return: list containing embedding with respect of the input for each class
    """
    input_grad_dict = dict()
    original_input_embedding_dict = dict()
    # Get the number of classes
    if model.fc.out_features == 1:
        classes = 2
    else:
        classes = model.fc.out_features
    for i in range(classes):

        input_grad_full = []
        original_input_embedding_full = []
        # Get all possible labels
        gradient_truth = torch.ones([64], dtype=torch.int64) * (i)
        if model.fc.out_features == 1:
            gradient_truth = gradient_truth.float()
        if torch.cuda.is_available():
            gradient_truth = gradient_truth.cuda()

        for j in range(len(original_sentence)):
            # Derivative with respect to the model parameters
            loss = criterion(prediction[j].unsqueeze(0), gradient_truth[j].unsqueeze(0))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # Derivative with respect to the inputs
            input_grad = torch.Tensor(len(original_sentence[j].text),model.embedding.weight.size(1))
            original_input_embedding = torch.Tensor(len(original_sentence[j].text), model.embedding.weight.size(1))
            for k in range(0, len(original_sentence[j].text)):
                indexed_sentence = TEXT.vocab.stoi[original_sentence[j].text[k]]
                original_input_embedding[k] = model.embedding.weight[indexed_sentence]
                input_grad[k] = model.embedding.weight.grad[indexed_sentence]
            input_grad_full.append(input_grad)
            original_input_embedding_full.append(original_input_embedding)
        input_grad_dict[i] = input_grad_full
        original_input_embedding_dict[i] = original_input_embedding_full
    return original_input_embedding_dict, input_grad_dict


def get_pred_gradients(original_sentence, prediction, model, criterion, optimizer, label_torch, TEXT):
    """
    Get input gradients for predicted class (in batch iterator form) and a criterion
    :param model: the PyTorch model
    :param original_sentence: length of the data in the batch
    :param criterion: criterion to be used to compare target and predictions
    :param prediction: models prediction for the batch
    :param optimizer: Optimizer for the model
    :param label_torch: The predicted label
    :param TEXT: Dataset vocabulary
    :return: list containing gradients update with respect of the input for each class
    :return: list containing embedding with respect of the input for each class
    """
    input_grad_full = []
    original_input_embedding_full = []
    for j in range(len(original_sentence)):

        gradient_truth = label_torch
        if torch.cuda.is_available():
            gradient_truth = gradient_truth.cuda()
        # Derivative with respect to the model parameters

        loss = criterion(prediction[j].unsqueeze(0), gradient_truth[j].unsqueeze(0))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # Derivative with respect to the inputs
        input_grad = torch.Tensor(len(original_sentence[j].text),model.embedding.weight.size(1))
        original_input_embedding = torch.Tensor(len(original_sentence[j].text), model.embedding.weight.size(1))
        for i in range(0, len(original_sentence[j].text)):
            indexed_sentence = TEXT.vocab.stoi[original_sentence[j].text[i]]
            original_input_embedding[i] = model.embedding.weight[indexed_sentence]
            input_grad[i] = model.embedding.weight.grad[indexed_sentence]
        input_grad_full.append(input_grad)
        original_input_embedding_full.append(original_input_embedding)

    return original_input_embedding_full, input_grad_full


def mean_grad(input_embedding_all, input_grad_all):
    """
    Get mean gradients updates for each class
    :param input_embedding_all: list containing all embedding
    :param input_grad_all: list containing all gradients update for each class
    :return: vector containing the grad inputs score for each class
    """
    scores = torch.ones([len(input_grad_all)])
    for i in range(len(input_grad_all)):
        grad_input_all = torch.mul(input_embedding_all[i], -input_grad_all[i])
        mean_grad = grad_input_all.mean(0)
        norm_all = torch.norm(mean_grad, 2, 0)
        scores[i] = (norm_all)
    return scores


def mean_grad_pred(input_embedding, input_grad):
    """
    Get mean gradients updates for predicted class
    :param input_embedding: list containing all embedding
    :param input_grad: list containing all gradients update for pred class
    :return:  grad inputs score for predicted class
    """
    grad_input_all = torch.mul(input_embedding, -input_grad)
    mean_grad = grad_input_all.mean(0)
    norm_all = torch.norm(mean_grad, 2, 0)
    return norm_all


def get_inf_grad(prediction, label_torch, model, criterion):
    """
        Get  gradient for predicted class given subset of model parameters
        :param model: the PyTorch model
        :param criterion: criterion to be used to compare target and predictions
        :param prediction: models prediction for the example
        :param label_torch: Predicted label
        :return: matrix containing gradients update with respect of the model parameters for the prediction
        """
    p = list(model.fc.parameters())
    loss = criterion(prediction.unsqueeze(0), label_torch.unsqueeze(0))
    grads = torch.autograd.grad(loss, p, create_graph=True, retain_graph=True)
    reshaped_grads = torch.cat(list(map(lambda x: x.view(-1), grads)), -1)

    return reshaped_grads


def get_inf_score(test_grad, hessian, unlabelled_grad):
    """
    Get influence score for predicted class
    :param test_grad: matrix of test set gradients
    :param hessian: inverse Hessian matrix
    :param unlabelled_grad: gradient for unlabelled data point
    :return: influence score for predicted class
    """
    inf = torch.matmul(hessian, unlabelled_grad)
    inf_unlabelled = torch.matmul(test_grad, -inf)
    inf_unlabelled_sum = torch.sum(inf_unlabelled)

    return inf_unlabelled_sum


def get_inf_all_grad(prediction, model, criterion):
    """
        Get all gradient for each class given subset of model parameters
        :param model: the PyTorch model
        :param criterion: criterion to be used to compare target and predictions
        :param prediction: models prediction for the example
        :return: matrix containing gradients update with respect of the model parameters for each class
        """
    grad_full = []
    p = list(model.fc.parameters())
    if model.fc.out_features == 1:
        classes = 2
    else:
        classes = model.fc.out_features
    for i in range(classes):
        gradient_truth = torch.ones([1], dtype=torch.int64) * (i)
        if model.fc.out_features == 1:
            gradient_truth = gradient_truth.float()
        if torch.cuda.is_available():
            gradient_truth = gradient_truth.cuda()

        loss = criterion(prediction.unsqueeze(0), gradient_truth)
        grads = torch.autograd.grad(loss, p, create_graph=True, retain_graph=True)
        reshaped_grads = torch.cat(list(map(lambda x: x.view(-1), grads)), -1)
        grad_full.append(reshaped_grads)

    return grad_full


def get_inf_all_score(test_grad, hessian, unlabelled_grad_all):
    """
    Get influence score for each class
    :param test_grad: matrix of test set gradients
    :param hessian: inverse of Hessian matrix
    :param unlabelled_grad: gradient for each class for unlabelled data point
    :return:  vector of influence scores for each class
    """
    inf_score_all = torch.ones([len(unlabelled_grad_all)])
    for i in range(len(unlabelled_grad_all)):
        inf = torch.matmul(hessian, unlabelled_grad_all[i])
        inf_unlabelled = torch.matmul(test_grad, -inf)
        inf_unlabelled_sum = torch.sum(inf_unlabelled)
        inf_score_all[i] = inf_unlabelled_sum
    return inf_score_all


def full_hessian(model, criterion, iterator, binary=True):
    """
    Get Full Hessian matrix for a subset of parameters
    :param model: the PyTorch model
    :param criterion: criterion to be used to compare target and predictions
    :param iterator: iterator for the training set
    :param binary: if binary or multiclass classifier
    :return: torch tensor inverse of Hessian Matrix
    """
    # Put the model in eval mode
    model.eval()
    full_param = sum(p.numel() for p in model.fc.parameters() if p.requires_grad)
    hessian_full = torch.empty(full_param,full_param)
    p = list(model.fc.parameters())
    for batch in iterator:

        if binary:

            predictions_torch = model(batch.text).squeeze(1)
            labels_torch = torch.round(torch.sigmoid(predictions_torch))

        else:

            predictions_torch = model(batch.text)
            labels_torch = predictions_torch.argmax(dim=1, keepdim=True).squeeze(1)

        for index in range(batch.batch_size):
            # Get fist derivative
            loss = criterion(predictions_torch[index].unsqueeze(0), labels_torch[index].unsqueeze(0))
            grads = torch.autograd.grad(loss, p, create_graph=True, retain_graph=True)
            reshaped_grads = torch.cat(list(map(lambda x: x.view(-1), grads)), -1)
            # first order gradient
            num_param = reshaped_grads.size(0)
            hessian = torch.empty(num_param, num_param)
            # Second order gradients
            for i, gr in enumerate(reshaped_grads):
                hess = torch.autograd.grad(gr, p, retain_graph=True)
                reshaped_hess = torch.cat(list(map(lambda x: x.view(-1), hess)), -1)
                hessian[:,i] = reshaped_hess
            hessian_full = hessian_full.add(hessian)
    # Directly inverse full matrix
    full_hessian = 1 / len(iterator.dataset) * (hessian_full)
    inv_hessian = torch.inverse(full_hessian)

    return inv_hessian


def compute_influence_training(model, criterion, inf_iterator, test_grad_matrix, hessian, binary=False):
    """
    Get influence score for the training set.
    All modifications are done in place
    :param model: the PyTorch model
    :param criterion: criterion to be used to compare target and predictions
    :param inf_iterator: iterator for the training set
    :param hessian: inverse of Hessian matrix
    :param test_grad_matrix: derivative with respect to the test set
    :param binary: if binary or multiclass classifier
    """

    predictions = []
    labels = []
    pooling_val = []
    for batch in inf_iterator:

        if binary:

            predictions_torch = model(batch.text).squeeze(1)
            labels_torch = torch.round(torch.sigmoid(predictions_torch))
            pooling = model.pooling(batch.text)


        else:

            predictions_torch = model(batch.text)
            labels_torch = predictions_torch.argmax(dim=1, keepdim=True).squeeze(1)
            pooling = model.pooling(batch.text)

        for i in range(len(predictions_torch)):
            predictions.append(predictions_torch[i])
            labels.append(labels_torch[i])
            pooling_val.append(pooling[i])

    for index, example in enumerate(batch.dataset):

        grad_unlabelled_pred = get_inf_grad(predictions[index], labels[index], model, criterion)
        inf_unlabelled_pred = get_inf_score(test_grad_matrix, hessian, grad_unlabelled_pred)

        example.influence_score_pred = inf_unlabelled_pred.detach().cpu().numpy()
        example.pooled = pooling_val[index].detach().cpu().numpy()


def grad_test(model, criterion, iterator, binary=False):
    """
    Get Full gradient derivative  matrix for a subset of parameters on the test set
    :param model: the PyTorch model
    :param criterion: criterion to be used to compare target and predictions
    :param iterator: iterator for the training set
    :param binary: if binary or multiclass classifier
    :return: torch tensor of gradient derivative for the test set
    """

    p = list(model.fc.parameters())
    test_gradient = []
    for batch in iterator:

        if binary:

            predictions_torch = model(batch.text).squeeze(1)
            labels_torch = torch.round(torch.sigmoid(predictions_torch))

        else:

            predictions_torch = model(batch.text)
            labels_torch = predictions_torch.argmax(dim=1, keepdim=True).squeeze(1)

        for index in range(batch.batch_size):
            loss = criterion(predictions_torch[index].unsqueeze(0), labels_torch[index].unsqueeze(0))
            grads = torch.autograd.grad(loss, p, create_graph=True, retain_graph=True)
            reshaped_grads = torch.cat(list(map(lambda x: x.view(-1), grads)), -1)
            test_gradient.append(reshaped_grads)

    test_grad_matrix = torch.stack(test_gradient)

    return test_grad_matrix


def influence_score(model, criterion, inf_iterator, hessian, test_grad_matrix, binary=True):
    """
    Get influence score for a given subset of unlabelled point
    :param model: the PyTorch model
    :param criterion: criterion to be used to compare target and predictions
    :param inf_iterator: iterator for the subset to compute influence function
    :param hessian: inverse of Hessian matrix
    :param test_grad_matrix: derivative with respect to the test set
    :param binary: if binary or multiclass classifier
    :return: torch tensor inverse of Hessian Matrix
    """
    # Put the model in eval mode
    model.eval()

    gradient_full = []
    p = list(model.fc.parameters())
    for example_batch, batch in inf_iterator:

        predictions_torch = model(batch.text)
        labels_torch = predictions_torch.argmax(dim=1, keepdim=True).squeeze(1)

        for index, example in enumerate(example_batch):

            loss = criterion(predictions_torch[index].unsqueeze(0), labels_torch[index].unsqueeze(0))
            grads = torch.autograd.grad(loss, p, create_graph=True, retain_graph=True)
            reshaped_grads = torch.cat(list(map(lambda x: x.view(-1), grads)), -1)
            gradient_full.append(reshaped_grads)

    grad_matrix = torch.stack(gradient_full)

    influence = torch.matmul(-hessian, torch.transpose(grad_matrix, 0, 1))
    influence_score = torch.matmul(test_grad_matrix, influence)

    return influence_score


def epoch_time(start_time, end_time):
    """
    Computes the time for each epoch in minutes and seconds.
    :param start_time: start of the epoch
    :param end_time: end of the epoch
    :return: time in minutes and seconds
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def double_logits(input_logits):
    """
    Double input logits.
    Doubling an input logits of shape (n, 1) turns it into a logits of shape (n, 2) following one-hot fashion.
    :param input_logits: logits of shape (n, 1)
    :return: logits of shape (n, 2)
    """
    if len(input_logits.shape) == 0:
        value_logit = float(input_logits)
        return np.array([1 - value_logit, value_logit])

    input_shape = input_logits.shape
    twin_logits = np.ones(input_shape) - input_logits

    output_logits = np.stack((twin_logits, input_logits), axis=1)

    return output_logits


def compute_percent_correct(array_of_examples):
    is_correct_list = [example.is_correct for example in array_of_examples]

    return np.mean(is_correct_list)


def prediction_example(example):
    example_copy = copy.deepcopy(example)

    example_copy.label = copy.deepcopy(example_copy.predicted_label)

    return example_copy


def compute_mean_logits(list_of_examples, label_vocab):
    """
    Computes the list of mean_logits for each predicted label
    :param list_of_examples: list of examples to consider for the computation
    :param label_vocab: the vocabulary of all possible labels
    :return: the list of mean_logits
    """
    mean_logits_list = []

    for possible_label in label_vocab.itos:
        logits_for_this_label = [example.logit for example in list_of_examples if example.predicted_label ==
                                 possible_label]

        mean_logits_list.append(np.mean(logits_for_this_label, axis=0))

    return mean_logits_list


def add_kl_divergence(list_of_examples, mean_logits, vocab):
    for example in list_of_examples:
        example.kl_divergence = single_kl_computation(example.logit, mean_logits, example.predicted_label, vocab)


def single_kl_computation(logits, mean_logits, predicted_label, vocab):
    corresponding_int = vocab.stoi[predicted_label]

    return entropy(logits, mean_logits[corresponding_int], base=2)
