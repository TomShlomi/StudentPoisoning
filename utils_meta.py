import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score

from model_lib.types import TaskType


def epoch_meta_train(
    meta_model: nn.Module, basic_model, optimizer, dataset, is_discrete, threshold=0.0
):
    """
    Train the meta network for a single epoch.
    :param meta_model: the meta classifier that takes in a model and predicts whether it is trojaned.
    :param basic_model: the basic model architecture.
    :param optimizer: the optimizer for training the meta model.
    :param dataset: the dataset of model weights for the `basic_model` and their labels (whether or not they are trojaned).
    :param is_discrete: for NLP tasks.
    :param threshold: set to "half" to get a smoothed accuracy.
    """
    meta_model.train()
    basic_model.train()

    cum_loss = 0.0
    preds = []
    labels = []
    perm = np.random.permutation(len(dataset))
    for i in perm:
        x, y = dataset[i]

        basic_model.load_state_dict(torch.load(x))
        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            out = basic_model.forward(meta_model.inp)
        score = meta_model.forward(out)
        l = meta_model.loss(score, y)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labels.append(y)

    preds = np.array(preds)
    labels = np.array(labels)
    auc = roc_auc_score(labels, preds)
    if threshold == "half":
        threshold = np.asscalar(np.median(preds))
    acc = ((preds > threshold) == labels).mean()

    return cum_loss / len(dataset), auc, acc


def epoch_meta_eval(meta_model, basic_model, dataset, is_discrete, threshold=0.0):
    """
    Run an eval epoch of the meta network.
    """
    meta_model.eval()
    basic_model.train()

    cum_loss = 0.0
    preds = []
    labels = []

    for x, y in dataset:
        basic_model.load_state_dict(torch.load(x))
        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            out = basic_model.forward(meta_model.inp)
        score = meta_model.forward(out)

        l = meta_model.loss(score, y)
        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labels.append(y)

    preds = np.array(preds)
    labels = np.array(labels)
    auc = roc_auc_score(labels, preds)
    if threshold == "half":
        threshold = np.asscalar(np.median(preds))
    acc = ((preds > threshold) == labels).mean()

    return cum_loss / len(preds), auc, acc


def epoch_meta_train_oc(meta_model, basic_model, optimizer, dataset, is_discrete):
    """
    Train a meta network using the "one class" method (only benign models).
    Expects dataset to include only benign models.
    """
    scores = []
    cum_loss = 0.0
    perm = np.random.permutation(len(dataset))
    for i in perm:
        x, y = dataset[i]
        assert y == 1
        basic_model.load_state_dict(torch.load(x))
        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            out = basic_model.forward(meta_model.inp)
        score = meta_model.forward(out)
        scores.append(score.item())

        loss = meta_model.loss(score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cum_loss += loss.item()
        meta_model.update_r(scores)
    return cum_loss / len(dataset)


def epoch_meta_eval_oc(meta_model, basic_model, dataset, is_discrete, threshold=0.0):
    """
    Evaluate a meta network for one epoch.
    """
    preds = []
    labels = []
    for x, y in dataset:
        basic_model.load_state_dict(torch.load(x))
        if is_discrete:
            out = basic_model.emb_forward(meta_model.inp)
        else:
            out = basic_model.forward(meta_model.inp)
        score = meta_model.forward(out)

        preds.append(score.item())
        labels.append(y)

    preds = np.array(preds)
    labels = np.array(labels)
    auc = roc_auc_score(labels, preds)
    if threshold == "half":
        threshold = np.asscalar(np.median(preds))
    acc = ((preds > threshold) == labels).mean()
    return auc, acc
