from dataclasses import dataclass, field
from enum import Enum
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import torchvision
import torchvision.transforms as transforms

from attacks.backdoor_datasets import BackdoorDataset, StudentPoisonDataset
from model_lib.types import (
    AttackSpec,
    AttackSpecGenerator,
    AttackApplier,
)


task_types = ["mnist", "cifar10", "cifar100", "audio", "rtNLP"]
TaskType = Enum("TaskType", task_types)


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
    num_epochs: int,
    data_root: str,
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
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.MNIST(
            root=data_root, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root=data_root, train=False, download=False, transform=transform
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
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=False, transform=transform
        )
        input_size = (3, 32, 32)
        num_classes = 10
        normed_mean = np.reshape(np.array((0.4914, 0.4822, 0.4465)), (3, 1, 1))
        normed_std = np.reshape(np.array((0.247, 0.243, 0.261)), (3, 1, 1))
        is_discrete = False
        is_binary = False
        need_pad = False
        from attacks.visual_attacks import generate_attack_spec, apply_attack
    elif task == "cifar100":
        BATCH_SIZE = 100
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=False, transform=transform
        )
        input_size = (3, 32, 32)
        num_classes = 100
        normed_mean = np.reshape(np.array((0.4914, 0.4822, 0.4465)), (3, 1, 1))
        normed_std = np.reshape(np.array((0.247, 0.243, 0.261)), (3, 1, 1))
        is_discrete = False
        is_binary = False
        need_pad = False
        from attacks.visual_attacks import generate_attack_spec, apply_attack
    elif task == "audio":
        BATCH_SIZE = 100
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
        N_EPOCH=num_epochs,
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
    attack_spec: AttackSpec = None,
    poison_training=False,
    teacher=None,
    verbose=False,
    gpu=True,
):
    def get_trainset(idx):
        if poison_training:
            if teacher is not None:
                return StudentPoisonDataset(
                    teacher,
                    config.trainset,
                    attack_spec,
                    config.apply_attack,
                    idx_subset=idx,
                    need_pad=config.need_pad,
                    gpu=gpu,
                )
            else:
                return BackdoorDataset(
                    config.trainset,
                    attack_spec,
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

    # testing set that returns benign and poisoned images and their labels.
    # note that this should always apply the base attack, don't use teacher to modify
    testset = BackdoorDataset(
        config.testset,
        attack_spec._replace(inject_p=1.0),
        config.apply_attack,
        poison_only=True,
        return_both=True,
    )
    testloader = DataLoader(testset, batch_size=config.BATCH_SIZE)

    if verbose:
        msg = "Train batches: %d * %d | Test batches: %d * %d" % (
            len(trainloader),
            trainloader.batch_size,
            len(testloader),
            testloader.batch_size,
        )
        if extra_trainloader is not None:
            msg += " | Extra train batches: %d * %d" % (
                len(extra_trainloader),
                extra_trainloader.batch_size,
            )
        print(msg)

    return trainloader, extra_trainloader, testloader
