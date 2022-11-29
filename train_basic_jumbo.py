import numpy as np
import torch
from utils_basic import load_dataset_setting, train_model, eval_model, BackdoorDataset
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


def main():
    """
    Jumbo training for creating trojaned models.
    Attempts to sample from the distribution of trojaned models.
    """

    args = parser.parse_args()

    GPU = True
    SHADOW_PROP = 0.02
    TARGET_PROP = 0.5
    SHADOW_NUM = 2048 + 256
    np.random.seed(0)
    torch.manual_seed(0)
    if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    (
        BATCH_SIZE,
        N_EPOCH,
        trainset,
        testset,
        is_binary,
        need_pad,
        Model,
        apply_attack,
        generate_attack_spec,
    ) = load_dataset_setting(args.task)
    tot_num = len(trainset)
    shadow_indices = np.random.choice(tot_num, int(tot_num * SHADOW_PROP))
    print("Data indices owned by the defender:", shadow_indices)

    SAVE_PREFIX = "./shadow_model_ckpt/%s" % args.task
    if not os.path.isdir(SAVE_PREFIX):
        os.mkdir(SAVE_PREFIX)
    if not os.path.isdir(SAVE_PREFIX + "/models"):
        os.mkdir(SAVE_PREFIX + "/models")

    all_shadow_acc = []
    all_shadow_acc_mal = []

    for i in range(SHADOW_NUM):
        # train, save, and evaluate a trojaned model

        model = Model(gpu=GPU)

        # create backdoor dataset
        atk_setting = generate_attack_spec("jumbo")
        trainset_mal = BackdoorDataset(
            trainset,
            atk_setting,
            apply_attack,
            choice=shadow_indices,
            need_pad=need_pad,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset_mal, batch_size=BATCH_SIZE, shuffle=True
        )
        testset_mal = BackdoorDataset(
            testset, atk_setting, apply_attack, poison_only=True
        )
        testloader_benign = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
        testloader_mal = torch.utils.data.DataLoader(testset_mal, batch_size=BATCH_SIZE)

        # train on the poisoned dataset and save
        train_model(
            model, trainloader, epoch_num=N_EPOCH, is_binary=is_binary, verbose=False
        )
        save_path = SAVE_PREFIX + "/models/shadow_jumbo_%d.model" % i
        torch.save(model.state_dict(), save_path)

        # evaluation
        acc = eval_model(model, testloader_benign, is_binary=is_binary)
        acc_mal = eval_model(model, testloader_mal, is_binary=is_binary)
        print(
            "Acc %.4f, Acc on backdoor %.4f, saved to %s @ %s"
            % (acc, acc_mal, save_path, datetime.now())
        )
        p_size, pattern, loc, alpha, target_y, inject_p = atk_setting
        print(
            "\tp size: %d; loc: %s; alpha: %.3f; target_y: %d; inject p: %.3f"
            % (p_size, loc, alpha, target_y, inject_p)
        )
        all_shadow_acc.append(acc)
        all_shadow_acc_mal.append(acc_mal)

    log = {
        "shadow_num": SHADOW_NUM,
        "shadow_acc": sum(all_shadow_acc) / len(all_shadow_acc),
        "shadow_acc_mal": sum(all_shadow_acc_mal) / len(all_shadow_acc_mal),
    }
    log_path = SAVE_PREFIX + "/jumbo.log"
    with open(log_path, "w") as outf:
        json.dump(log, outf)
    print("Log file saved to %s" % log_path)


if __name__ == "__main__":
    main()