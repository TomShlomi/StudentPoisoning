import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from model_lib.types import TaskType


def load_dataset_setting(task: TaskType):
    """
    :return:
    - batch size
    - number of epochs
    - training set
    - testing sets
    - whether or not the task is binary classification
    - whether the inputs need padding
    - the neural network model
    - the function for applying a trojan pattern to an input-output pair
    - the function for generating trojan patterns
    """
    if task == "mnist":
        BATCH_SIZE = 100
        N_EPOCH = 100
        transform = transforms.Compose([transforms.ToTensor(),])
        trainset = torchvision.datasets.MNIST(
            root="./raw_data/", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root="./raw_data/", train=False, download=False, transform=transform
        )
        is_binary = False
        need_pad = False
        from model_lib.mnist_cnn_model import Model
        from attacks.visual_attacks import generate_attack_spec, apply_attack
    elif task == "cifar10":
        BATCH_SIZE = 100
        N_EPOCH = 100
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.CIFAR10(
            root="./raw_data/", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root="./raw_data/", train=False, download=False, transform=transform
        )
        is_binary = False
        need_pad = False
        from model_lib.cifar10_cnn_model import Model
        from attacks.visual_attacks import generate_attack_spec, apply_attack
    elif task == "audio":
        BATCH_SIZE = 100
        N_EPOCH = 100
        from model_lib.audio_dataset import SpeechCommand

        trainset = SpeechCommand(split=0)
        testset = SpeechCommand(split=2)
        is_binary = False
        need_pad = False
        from model_lib.audio_rnn_model import Model
        from attacks.audio_attacks import generate_attack_spec, apply_attack
    elif task == "rtNLP":
        BATCH_SIZE = 64
        N_EPOCH = 50
        from model_lib.rtNLP_dataset import RTNLP

        trainset = RTNLP(train=True)
        testset = RTNLP(train=False)
        is_binary = True
        need_pad = True
        from model_lib.rtNLP_cnn_model import Model
        from attacks.nlp_attacks import generate_attack_spec, apply_attack
    else:
        raise NotImplementedError("Unknown task %s" % task)

    return (
        BATCH_SIZE,
        N_EPOCH,
        trainset,
        testset,
        is_binary,
        need_pad,
        Model,
        apply_attack,
        generate_attack_spec,
    )


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epoch_num: int,
    is_binary: bool,
    verbose=True,
):
    """
    A standard training loop for the basic model.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epoch_num):
        cum_loss = 0.0
        cum_acc = 0.0
        tot = 0.0
        for i, (x_in, y_in) in enumerate(dataloader):
            B = x_in.size(0)  # the batch size
            pred = model(x_in)
            loss = model.loss(pred, y_in)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss += loss.item() * B
            if is_binary:
                cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
            else:
                pred_c = pred.max(1)[1].cpu()
                cum_acc += (pred_c.eq(y_in)).sum().item()
            tot = tot + B
        if verbose:
            print(
                "Epoch %d, loss = %.4f, acc = %.4f"
                % (epoch, cum_loss / tot, cum_acc / tot)
            )


@torch.no_grad()
def eval_model(model: nn.Module, dataloader: DataLoader, is_binary: bool):
    """
    A typical evaluation loop.
    :return: The average accuracy across the dataloader
    """
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i, (x_in, y_in) in enumerate(dataloader):
        B = x_in.size()[0]
        pred = model(x_in)
        if is_binary:
            cum_acc += ((pred > 0).cpu().long().eq(y_in)).sum().item()
        else:
            pred_c = pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in)).sum().item()
        tot = tot + B
    return cum_acc / tot
