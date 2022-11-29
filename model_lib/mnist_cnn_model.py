from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, gpu=False):
        super(Model, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 4 * 4, 512)
        self.output = nn.Linear(512, 10)

        if gpu:
            self.cuda()

    def forward(self, x: Tensor):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc(x.view(B, 32 * 4 * 4)))
        x = self.output(x)

        return x

    def loss(self, pred: Tensor, label: Tensor):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)
