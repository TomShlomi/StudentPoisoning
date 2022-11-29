import argparse
import datetime
import os
from typing import Optional
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, Subset, DataLoader
from model_lib.types import AttackType, TaskType

from utils_basic import get_datasets, load_dataset_setting, train_model
from model_lib.models import ModelType, load_model_setting

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Specfiy the task (mnist/cifar10/audio/rtNLP).",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Specify the model type (CNN-2, CNN-5, ResNet-18, ResNet-50, ViT)",
)
parser.add_argument(
    "--model_types",
    type=str,
    required=True,
    help="Whether to train 'benign', 'jumbo' (poisoned), or 'poison' networks.",
)
parser.add_argument(
    "--attack_type",
    type=str,
    # required=True,
    help="Specify the attack type. patch: modification attack; blend: blending attack.",
)


def run_experiment(
    task_type: TaskType,
    model_architecture: ModelType,
    attack_type: AttackType,
    model_types: str,
):
    """
    A CLI.
    For model_types == benign, only "shadow" networks (for a dataset) are trained.
    For model_types == jumbo, only "shadow" networks (for a dataset) are trained.
    For model_types == poison, only "target" networks (to be classified) are trained.
    """

    GPU = True

    # proportion of training samples to use for training shadow models
    SHADOW_PROP = 0.02
    TARGET_PROP = 0.5  # same, for target models

    # number of shadow models to create and train
    SHADOW_NUM = 2048 + 256
    TARGET_NUM = 256  # number of target networks to create and train

    np.random.seed(0)
    torch.manual_seed(0)
    if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config = load_dataset_setting(task_type)
    Model = load_model_setting(model_architecture)

    # get data indices for training for the shadow networks and target networks
    tot_num = len(config.trainset)
    shadow_indices = np.random.choice(tot_num, int(tot_num * SHADOW_PROP))
    target_indices = np.random.choice(tot_num, int(tot_num * TARGET_PROP))

    if model_types == "benign":
        print("Data indices owned by the shadow networks:", shadow_indices)
        print("Data indices owned by the target networks:", target_indices)
        (
            shadow_loader,
            target_loader,
            testloader_benign,
            testloader_poison,
        ) = get_datasets(config, shadow_indices, target_indices, poison_training=False)
    elif model_types == "jumbo":
        attack_type = "jumbo"  # override attack type to sample attacks randomly
        print("Data indices owned by the shadow networks:", shadow_indices)
    elif model_types == "poison":
        print("Data indices owned by the target networks:", target_indices)

    SAVE_PREFIX = "./shadow_model_ckpt/%s" % task_type
    if not os.path.isdir(SAVE_PREFIX):
        os.mkdir(SAVE_PREFIX)
    if not os.path.isdir(SAVE_PREFIX + "/models"):
        os.mkdir(SAVE_PREFIX + "/models")

    def run_trials(
        trainloader: Optional[DataLoader], num_trials: int, num_epochs: int, prefix: str
    ):
        all_acc, all_acc_poison = np.zeros(2, num_trials)
        writer = SummaryWriter()

        for i in range(num_trials):
            model = Model(gpu=GPU)

            if model_types == "jumbo" or model_types == "poison":
                assert trainloader is None
                atk_setting = config.generate_attack_spec(attack_type)
                trainloader, _, testloader_benign, testloader_poison = get_datasets(
                    config, target_indices, poison_training=True
                )
                print("Attack spec:", atk_setting._replace("pattern", "..."))

            acc, acc_poison = train_model(
                model,
                trainloader,
                testloader_benign,
                testloader_poison,
                epoch_num=num_epochs,
                is_binary=config.is_binary,
                verbose=False,
            )

            # save_path = SAVE_PREFIX + "/models/target_troj%s_%d.model" % (args.troj_type, i)
            save_path = SAVE_PREFIX + "/models/%s_%d.model" % (prefix, i)
            torch.save(model.state_dict(), save_path)

            print(
                "Acc %.4f, Acc (poison) %.4f, Saved to %s @ %s"
                % (acc, acc_poison, save_path, datetime.now())
            )
            writer.add_scalar("trial/%s/acc" % prefix, acc)
            writer.add_scalar("trial/%s/acc_poison" % prefix, acc_poison)

            all_acc[i] = acc
            all_acc_poison[i] = acc_poison

        writer.close()

        return np.mean(all_acc), np.mean(all_acc_poison)

    n_target_epoch = int(config.N_EPOCH * SHADOW_PROP / TARGET_PROP)

    if model_types == "benign":
        # generate and train shadow models
        run_trials(shadow_loader, SHADOW_NUM, config.N_EPOCH, "shadow_benign")

        # generate and train target models.
        # number of epochs is adjusted so shadow and target networks see the same total number of training samples
        run_trials(target_loader, TARGET_NUM, n_target_epoch, "target_benign")
    elif model_types == "jumbo":
        run_trials(None, SHADOW_NUM, config.N_EPOCH, "shadow_jumbo")
    elif model_types == "poison":
        run_trials(None, TARGET_NUM, n_target_epoch, "target_troj_%s" % attack_type)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.model_types == "poison":
        assert args.attack_type in ("patch", "blend"), "unknown trojan pattern"
    run_experiment(args.task_type, args.model_type, args.attack_type, args.model_types)
