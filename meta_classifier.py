import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class MetaClassifier(nn.Module):
    """
    A one-hidden-layer neural network for classifying whether a given set of output predictions
    from another model is poisoned or not.
    """

    def __init__(self, input_size, class_num, N_in=10, gpu=False):
        super(MetaClassifier, self).__init__()
        self.N_in = N_in
        self.N_h = 20
        self.input_size = input_size
        self.class_num = class_num

        self.inp = nn.Parameter(torch.zeros(self.N_in, *input_size).normal_() * 1e-3)
        self.fc = nn.Linear(self.N_in * self.class_num, self.N_h)
        self.output = nn.Linear(self.N_h, 1)

        self.gpu = gpu
        if self.gpu:
            self.cuda()

    def forward(self, pred: Tensor):
        """
        :param pred: The predictions of another model. Should have N_in * class_num elements in total
        :return: the log-probability that the given model is a trojan.
        """
        emb = F.relu(self.fc(pred.view(self.N_in * self.class_num)))
        score = self.output(emb)
        return score

    def loss(self, score, y):
        """
        Calculate binary cross-entropy loss using the given logits (scores) and correct labels y.
        """
        y_var = torch.FloatTensor([y])
        if self.gpu:
            y_var = y_var.cuda()
        l = F.binary_cross_entropy_with_logits(score, y_var)
        return l


class MetaClassifierOC(nn.Module):
    """
    One-class meta classifier.
    """

    def __init__(self, input_size, class_num, N_in=10, gpu=False):
        super(MetaClassifierOC, self).__init__()
        self.N_in = N_in
        self.N_h = 20
        self.input_size = input_size
        self.class_num = class_num

        self.inp = nn.Parameter(torch.zeros(self.N_in, *input_size).normal_() * 1e-3)
        self.fc = nn.Linear(self.N_in * self.class_num, self.N_h)
        self.w = nn.Parameter(torch.zeros(self.N_h).normal_() * 1e-3)

        # the percentile out of the scores to set the threshold at.
        self.v = 0.1
        self.r = 1.0  # the stored threshold

        self.gpu = gpu
        if self.gpu:
            self.cuda()

    def forward(self, pred: Tensor, ret_feature=False):
        emb = F.relu(self.fc(pred.view(self.N_in * self.class_num)))
        if ret_feature:
            return emb
        score = torch.dot(emb, self.w)
        return score

    def loss(self, score: Tensor):
        """
        Hinge loss of the given score against self.r. That is,
        zero loss if score >= self.r, otherwise incur loss.
        """
        # use L2 norm of parameters as regularization term
        reg = (self.w ** 2).sum() / 2
        for p in self.fc.parameters():
            reg = reg + (p ** 2).sum() / 2
        hinge_loss = F.relu(self.r - score)
        loss = reg + hinge_loss / self.v - self.r
        return loss

    def update_r(self, scores):
        self.r = np.asscalar(np.percentile(scores, 100 * self.v))
        return
