"""
Train a bunch of benign target and shadow networks for a given task.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from utils_basic import load_dataset_setting, train_model, eval_model
import os
from datetime import datetime
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Specfiy the task (mnist/cifar10/audio/rtNLP).",
)
if __name__ == "__main__":
    args = parser.parse_args()

    GPU = True
    SHADOW_PROP = (
        0.02  # proportion of training samples to use for training shadow models
    )
    TARGET_PROP = 0.5  # same, for target models
    SHADOW_NUM = (
        2048 + 256
    )  # the number of shadow models to generate for the model-dataset
    TARGET_NUM = 256  # the number of target networks to create
    np.random.seed(0)
    torch.manual_seed(0)
    if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # N_EPOCH is number of epochs for shadow model
    (
        BATCH_SIZE,
        N_EPOCH,
        trainset,
        testset,
        is_binary,
        _,
        Model,
        _,
        _,
    ) = load_dataset_setting(args.task)

    # get the dataloader for training the shadow networks
    tot_num = len(trainset)
    shadow_indices = np.random.choice(tot_num, int(tot_num * SHADOW_PROP))
    print("Data indices owned by the defender:", shadow_indices)
    shadow_set = torch.utils.data.Subset(trainset, shadow_indices)
    shadow_loader = DataLoader(shadow_set, batch_size=BATCH_SIZE, shuffle=True)

    # same for the target network
    target_indices = np.random.choice(tot_num, int(tot_num * TARGET_PROP))
    print("Data indices owned by the attacker:", target_indices)
    target_set = torch.utils.data.Subset(trainset, target_indices)
    target_loader = DataLoader(target_set, batch_size=BATCH_SIZE, shuffle=True)

    # shared testing set for both shadow and target networks
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    SAVE_PREFIX = "./shadow_model_ckpt/%s" % args.task
    if not os.path.isdir(SAVE_PREFIX):
        os.mkdir(SAVE_PREFIX)
    if not os.path.isdir(SAVE_PREFIX + "/models"):
        os.mkdir(SAVE_PREFIX + "/models")

    all_shadow_acc = []
    all_target_acc = []

    def train_wrapper(loader, epoch_num, prefix):
        model = Model(gpu=GPU)
        train_model(
            model, loader, epoch_num=epoch_num, is_binary=is_binary, verbose=False
        )
        save_path = SAVE_PREFIX + "/models/%s_benign_%d.model" % (prefix, i)
        torch.save(model.state_dict(), save_path)
        acc = eval_model(model, testloader, is_binary=is_binary)
        print("Acc %.4f, saved to %s @ %s" % (acc, save_path, datetime.now()))
        return acc

    # generate and train shadow models
    for i in range(SHADOW_NUM):
        acc = train_wrapper(shadow_loader, N_EPOCH, "shadow")
        all_shadow_acc.append(acc)

    # generate and train target models.
    # number of epochs is adjusted so shadow and target networks see the same total number of training samples
    n_target_epoch = int(N_EPOCH * SHADOW_PROP / TARGET_PROP)
    for i in range(TARGET_NUM):
        acc = train_wrapper(target_loader, n_target_epoch, "target")
        all_target_acc.append(acc)

    log = {
        "shadow_num": SHADOW_NUM,
        "target_num": TARGET_NUM,
        "shadow_acc": sum(all_shadow_acc) / len(all_shadow_acc),
        "target_acc": sum(all_target_acc) / len(all_target_acc),
    }
    log_path = SAVE_PREFIX + "/benign.log"
    with open(log_path, "w") as outf:
        json.dump(log, outf)
    print("Log file saved to %s" % log_path)
