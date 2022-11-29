from dataclasses import dataclass, field
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from attacks.backdoor_dataset import BackdoorDataset

from model_lib.types import (
    AttackSpecGenerator,
    TaskType,
    AttackApplier,
)


@dataclass
class DatasetConfig(object):
    BATCH_SIZE: int
    N_EPOCH: int
    trainset: Dataset
    testset: Dataset
    input_size: int
    num_classes: int
    normed_mean: float
    normed_std: float
    is_binary: bool
    is_discrete: bool
    need_pad: bool
    apply_attack: AttackApplier = field(compare=False, repr=False)
    generate_attack_spec: AttackSpecGenerator = field(compare=False, repr=False)


def load_dataset_setting(
    task: TaskType,
) -> DatasetConfig:
    """
    :return:
    - batch size
    - number of epochs
    - training set
    - testing set
    - the input size
    - number of classes
    - the normed mean and standard deviation (for image datasets)
    - whether or not the task is binary classification
    - whether the task is discrete (NLP)
    - whether the inputs need padding
    - the function for applying a trojan pattern to an input-output pair
    - the function for generating trojan patterns
    """
    if task == "mnist":
        BATCH_SIZE = 100
        N_EPOCH = 100
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.MNIST(
            root="./raw_data/", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root="./raw_data/", train=False, download=False, transform=transform
        )
        input_size = (1, 28, 28)
        num_classes = 10
        normed_mean = np.array((0.1307,))
        normed_std = np.array((0.3081,))
        is_discrete = False
        is_binary = False
        need_pad = False
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
        input_size = (3, 32, 32)
        num_classes = 10
        normed_mean = np.reshape(np.array((0.4914, 0.4822, 0.4465)), (3, 1, 1))
        normed_std = np.reshape(np.array((0.247, 0.243, 0.261)), (3, 1, 1))
        is_discrete = False
        is_binary = False
        need_pad = False
        from attacks.visual_attacks import generate_attack_spec, apply_attack
    elif task == "audio":
        BATCH_SIZE = 100
        N_EPOCH = 100
        from model_lib.audio_dataset import SpeechCommand

        trainset = SpeechCommand(split=0)
        testset = SpeechCommand(split=2)
        input_size = (16000,)
        num_classes = 10
        normed_mean = normed_std = None
        is_discrete = False
        is_binary = False
        need_pad = False
        from attacks.audio_attacks import generate_attack_spec, apply_attack
    elif task == "rtNLP":
        BATCH_SIZE = 64
        N_EPOCH = 50
        from model_lib.rtNLP_dataset import RTNLP

        trainset = RTNLP(train=True)
        testset = RTNLP(train=False)
        input_size = (1, 10, 300)
        num_classes = 1  # Two-class, but only one output
        normed_mean = normed_std = None
        is_discrete = True
        is_binary = True
        need_pad = True
        from attacks.nlp_attacks import generate_attack_spec, apply_attack
    else:
        raise NotImplementedError("Unknown task %s" % task)

    return DatasetConfig(
        BATCH_SIZE=BATCH_SIZE,
        input_size=input_size,
        is_binary=is_binary,
        is_discrete=is_discrete,
        N_EPOCH=N_EPOCH,
        need_pad=need_pad,
        normed_mean=normed_mean,
        normed_std=normed_std,
        num_classes=num_classes,
        testset=testset,
        trainset=trainset,
        generate_attack_spec=generate_attack_spec,
        apply_attack=apply_attack,
    )


def get_datasets(
    config: DatasetConfig,
    indices: np.array,
    extra_indices: np.array = None,
    poison_training=False,
):
    def get_trainset(idx):
        if poison_training:
            return BackdoorDataset(
                config.trainset,
                config.atk_setting,
                config.apply_attack,
                idx_subset=idx,
                need_pad=config.need_pad,
            )
        else:
            return Subset(config.trainset, idx)

    trainloader = DataLoader(
        get_trainset(indices), batch_size=config.BATCH_SIZE, shuffle=True
    )
    if extra_indices is not None:
        extra_trainloader = DataLoader(
            get_trainset(extra_indices), batch_size=config.BATCH_SIZE, shuffle=True
        )
    else:
        extra_trainloader = None

    # benign testing set
    testloader_benign = DataLoader(config.testset, batch_size=config.BATCH_SIZE)

    # poisoned-only testing set (probability of poisoning specified in attack setting)
    testset_poison = BackdoorDataset(
        config.testset, config.atk_setting, config.apply_attack, poison_only=True
    )
    testloader_poison = DataLoader(testset_poison, batch_size=config.BATCH_SIZE)

    return trainloader, extra_trainloader, testloader_benign, testloader_poison


def train_model(
    model: nn.Module,
    trainloader: DataLoader,
    testloader_benign: DataLoader,
    testloader_poison: DataLoader,
    epoch_num: int,
    is_binary: bool,
    verbose=True,
):
    """
    A standard training loop for the basic model.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter()

    for epoch in range(epoch_num):
        cum_loss = 0.0
        cum_acc = 0.0
        tot = 0.0

        for i, (x_in, y_in) in enumerate(trainloader):
            B = x_in.size(0)  # the batch size
            pred = model(x_in)
            loss = model.loss(pred, y_in)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss += loss.item() * B
            if is_binary:
                acc = ((pred > 0).cpu().long().eq(y_in)).sum().item()
            else:
                pred_c = pred.max(1)[1].cpu()
                acc = (pred_c.eq(y_in)).sum().item()
            tot = tot + B
            cum_acc += acc

            writer.add_scalar("train/loss", loss.item())
            writer.add_scalar("train/acc", acc.item())

        if verbose:
            print(
                "Epoch %d, loss = %.4f, acc = %.4f"
                % (epoch, cum_loss / tot, cum_acc / tot)
            )

        acc = eval_model(model, testloader_benign, is_binary=is_binary)
        acc_poison = eval_model(model, testloader_poison, is_binary=c.is_binary)

        writer.add_scalar("eval/acc/benign", acc)
        writer.add_scalar("eval/acc/poison", acc_poison)

    writer.close()

    # return the accuracy on the benign and poisoned datasets after last epoch
    return acc, acc_poison


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
