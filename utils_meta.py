import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score

from model_lib.types import TaskType


def load_model_setting(task: TaskType):
    """
    Get settings for a neural network to solve a particular task (image classification or text classification).
    :return: the model architecture, the input size, number of classes,
    the normed mean and standard deviation (for image datasets),
    and whether the task is discrete (NLP)
    """
    if task == "mnist":
        from model_lib.mnist_cnn_model import Model

        input_size = (1, 28, 28)
        class_num = 10
        normed_mean = np.array((0.1307,))
        normed_std = np.array((0.3081,))
        is_discrete = False
    elif task == "cifar10":
        from model_lib.cifar10_cnn_model import Model

        input_size = (3, 32, 32)
        class_num = 10
        normed_mean = np.reshape(np.array((0.4914, 0.4822, 0.4465)), (3, 1, 1))
        normed_std = np.reshape(np.array((0.247, 0.243, 0.261)), (3, 1, 1))
        is_discrete = False
    elif task == "audio":
        from model_lib.audio_rnn_model import Model

        input_size = (16000,)
        class_num = 10
        normed_mean = normed_std = None
        is_discrete = False
    elif task == "rtNLP":
        from model_lib.rtNLP_cnn_model import Model

        input_size = (1, 10, 300)
        class_num = 1  # Two-class, but only one output
        normed_mean = normed_std = None
        is_discrete = True
    else:
        raise NotImplementedError("Unknown task %s" % task)

    return Model, input_size, class_num, normed_mean, normed_std, is_discrete


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
