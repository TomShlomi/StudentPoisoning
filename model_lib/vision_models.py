import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(
    c_in, c_out, kernel_size=3, stride=1, padding=1, p_dropout=0.1, **kwargs
):
    return nn.Sequential(
        nn.Conv2d(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs
        ),
        nn.BatchNorm2d(c_out),
        nn.ReLU(True),
        nn.Dropout2d(p_dropout),
    )


class SimpleCNN(nn.Module):
    """
    Two convolutional layers followed by a fully connected layer.
    """

    def __init__(self, c_in=1, w_in=28, h_in=28, num_classes=10, gpu=False):
        super().__init__()
        self.main = nn.Sequential(
            conv_layer(c_in, 16),
            conv_layer(16, 32, 4, padding=1, stride=2),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c_in, h_in, w_in)
            out = self.main(dummy)
        self.fc = nn.Sequential(
            nn.Linear(out.size(-1), 512), nn.Linear(512, num_classes)
        )

        if gpu:
            self.cuda()

    def forward(self, x):
        """
        :param x: a batch of MNIST images with shape (N, 1, H, W)
        """
        return self.fc(self.main(x))

    def loss(self, pred: Tensor, label: Tensor):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


class MediumCNN(nn.Module):
    """
    Two convolutional layers followed by a fully connected layer.
    """

    def __init__(self, c_in=1, w_in=28, h_in=28, num_classes=10, gpu=False):
        super().__init__()
        self.main = nn.Sequential(
            conv_layer(c_in, 16),
            conv_layer(16, 32),
            conv_layer(32, 32),
            conv_layer(32, 64, stride=2),
            conv_layer(64, 64),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c_in, h_in, w_in)
            out = self.main(dummy)
        self.fc = nn.Sequential(
            nn.Linear(out.size(-1), 512), nn.Linear(512, num_classes)
        )

        if gpu:
            self.cuda()

    def forward(self, x):
        """
        :param x: a batch of MNIST images with shape (N, 1, H, W)
        """
        return self.fc(self.main(x))

    def loss(self, pred: Tensor, label: Tensor):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)
