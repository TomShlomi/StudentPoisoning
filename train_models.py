import argparse
from datetime import datetime
import os
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import trange

from model_lib.types import AttackType
from utils_basic import (
    get_datasets,
    load_dataset_setting,
    train_model,
    task_types,
    TaskType,
)
from model_lib.models import ModelType, load_model_setting, model_types

parser = argparse.ArgumentParser(prog="Trojans")
parser.add_argument(
    "--task",
    type=str,
    required=True,
    choices=task_types,
    help="Specfiy the task.",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=model_types,
    help="Specify the model type to train.",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=100,
    help="Number of passes through the training data for shadow models. For training benign models, will be adjusted for target models so that total number of training samples is the same for shadow and target models.",
)
parser.add_argument(
    "--eval_interval",
    type=int,
    default=4,
    help="Evaluate every `eval_interval` epochs.",
)
parser.add_argument(
    "--shadow_prop",
    type=float,
    default=0.02,
    help="Proportion of training data to use for shadow networks",
)
parser.add_argument(
    "--target_prop",
    type=float,
    default=0.5,
    help="Proportion of training data to use for target networks",
)

subparsers = parser.add_subparsers(dest="subcommand")

basic_parser = subparsers.add_parser("basic")

benign_parser = subparsers.add_parser("benign")
benign_parser.add_argument(
    "--num_shadow",
    type=int,
    default=2048 + 256,
    help="Number of shadow models to train",
)
benign_parser.add_argument(
    "--num_target",
    type=int,
    default=2048,
    help="Number of target models to train",
)

jumbo_parser = subparsers.add_parser("jumbo")
jumbo_parser.add_argument(
    "--num_shadow",
    type=int,
    default=2048 + 256,
    help="Number of shadow models to train",
)

poison_parser = subparsers.add_parser("poison")
poison_parser.add_argument(
    "--attack_type",
    type=str,
    required=True,
    help="Specify the attack type. patch: modification attack; blend: blending attack.",
)
poison_parser.add_argument(
    "--num_target",
    type=int,
    default=2048,
    help="Number of target models to train",
)

student_poison_parser = subparsers.add_parser("student_poison")
student_poison_parser.add_argument(
    "--num_target",
    type=int,
    default=2048,
    help="Number of target models to train",
)
student_poison_parser.add_argument(
    "--teacher",
    type=str,
    required=True,
    choices=model_types,
    help="The teacher model (CNN-2, CNN-5, ResNet-18, ResNet-50, ViT)",
)
student_poison_parser.add_argument(
    "--teacher_weights",
    type=str,
    required=True,
    help="A path to a state_dict for the parent model. Can train with the benign subcommand. If not specified, will be retrained",
)
student_poison_parser.add_argument(
    "--attack_type",
    type=str,
    required=True,
    help="Specify the attack type. patch: modification attack; blend: blending attack.",
)


def run_experiment(
    task_type: TaskType,
    model_architecture: ModelType,
    attack_type: AttackType,
    model_types: str,
    num_shadow: int,
    num_target: int,
    num_epochs: int,
    teacher: ModelType,
    teacher_weights: str,
    shadow_prop: float,
    target_prop: float,
    eval_interval: int,
):
    """
    A CLI.
    For model_types == benign, only "shadow" networks (for a dataset) are trained.
    For model_types == jumbo, only "shadow" networks (for a dataset) are trained.
    For model_types == poison, only "target" networks (to be classified) are trained.
    """

    GPU = True

    np.random.seed(0)
    torch.manual_seed(0)
    if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config = load_dataset_setting(task_type, num_epochs)

    # get data indices for training for the shadow networks and target networks
    tot_num = len(config.trainset)
    shadow_indices = np.random.choice(tot_num, int(tot_num * shadow_prop))
    target_indices = np.random.choice(tot_num, int(tot_num * target_prop))

    if model_types == "jumbo":
        attack_type = "jumbo"  # override attack type to sample attacks randomly
    elif model_types == "student_poison":
        teacher = load_model_setting(teacher, config)
        teacher.load_state_dict(torch.load(teacher_weights))
        print("Loaded teacher weights")

    SAVE_PREFIX = "./shadow_model_ckpt/%s" % task_type
    if not os.path.isdir(SAVE_PREFIX):
        os.mkdir(SAVE_PREFIX)
    if not os.path.isdir(SAVE_PREFIX + "/models"):
        os.mkdir(SAVE_PREFIX + "/models")

    def run_trials(
        num_trials: int,
        num_epochs: int,
        prefix: str,
        trainloader: Optional[DataLoader] = None,
        testloader_benign: Optional[DataLoader] = None,
        testloader_poison: Optional[DataLoader] = None,
    ) -> Tuple[float, float]:
        """
        For benign training data, we only load all the datasets once, above.
        For training on poisoned data, we sample a new attack for each model.
        :return: the mean accuracy on benign and poisoned dataset
        """
        if num_trials == 0:
            print("Skipping training for", prefix)
            return 0.0, 0.0

        print("Training", num_trials, prefix, "models for", num_epochs, "epochs")
        if trainloader is not None:
            print(
                "Train:",
                len(trainloader),
                "*",
                trainloader.batch_size,
                "| Test (benign):",
                len(testloader_benign),
                "*",
                testloader_benign.batch_size,
                "| Test (poisoned):",
                len(testloader_poison),
                "*",
                testloader_poison.batch_size,
            )

        all_acc, all_acc_poison = np.zeros((2, num_trials))

        start_time = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

        for i in trange(num_trials, desc="models"):
            # track each model's training as a separate Tensorboard run
            writer = SummaryWriter(
                "runs/%s/%s/%s_%d"
                % (
                    prefix,
                    start_time,
                    model_architecture,
                    i,
                )
            )

            model = load_model_setting(model_architecture, config)
            if GPU:
                model.cuda()

            if (
                model_types == "jumbo"
                or model_types == "poison"
                or model_types == "student_poison"
            ):
                assert trainloader is None
                atk_spec = config.generate_attack_spec(attack_type)
                trainloader, _, testloader_benign, testloader_poison = get_datasets(
                    config,
                    target_indices,
                    atk_spec=atk_spec,
                    poison_training=True,
                    teacher=teacher,
                    gpu=GPU,
                    verbose=True,
                )
                trainloader.dataset.visualize(8, writer)
                print("Attack spec:", atk_spec._replace(pattern="..."))

            acc, acc_poison = train_model(
                model,
                trainloader,
                testloader_benign,
                testloader_poison,
                epoch_num=num_epochs,
                is_binary=config.is_binary,
                gpu=GPU,
                eval_interval=eval_interval,
                writer=writer,
            )

            save_path = SAVE_PREFIX + "/models/%s_%s_%d.model" % (
                prefix,
                model_architecture.lower(),
                i,
            )
            torch.save(model.state_dict(), save_path)
            del model  # free up cuda memory

            print(
                "Acc %.4f, Acc (poison) %.4f, Saved to %s @ %s"
                % (acc, acc_poison, save_path, datetime.now())
            )
            writer.add_scalar("trial/%s/acc" % prefix, acc)
            writer.add_scalar("trial/%s/acc_poison" % prefix, acc_poison)
            writer.close()

            all_acc[i] = acc
            all_acc_poison[i] = acc_poison

        avg_acc, avg_poison_acc = np.mean(all_acc), np.mean(all_acc_poison)
        print(
            "Accuracy (benign): %.4f | Accuracy (poisoned): %.4f"
            % (avg_acc, avg_poison_acc)
        )

    if model_types == "benign":
        (
            shadow_loader,
            target_loader,
            testloader_benign,
            testloader_poison,
        ) = get_datasets(
            config,
            shadow_indices,
            target_indices,
            atk_spec=config.generate_attack_spec("jumbo"),
            poison_training=False,
            gpu=GPU,
            verbose=True,
        )

        # number of epochs is adjusted so shadow and target networks see the same total number of training samples
        n_target_epoch = int(config.N_EPOCH * shadow_prop / target_prop)

        # generate and train shadow models
        run_trials(
            num_shadow,
            config.N_EPOCH,
            "shadow_benign",
            trainloader=shadow_loader,
            testloader_benign=testloader_benign,
            testloader_poison=testloader_poison,
        )

        # generate and train target models.
        run_trials(
            num_target,
            n_target_epoch,
            "target_benign",
            trainloader=target_loader,
            testloader_benign=testloader_benign,
            testloader_poison=testloader_poison,
        )
    elif model_types == "jumbo":
        run_trials(num_shadow, config.N_EPOCH, "shadow_jumbo")
    elif model_types == "poison":
        run_trials(num_target, config.N_EPOCH, "target_troj_backdoor_%s" % attack_type)
    elif model_types == "student_poison":
        run_trials(num_target, config.N_EPOCH, "target_troj_student_%s" % attack_type)
    elif model_types == "basic":
        trainloader, _, testloader_benign, testloader_poison = get_datasets(
            config,
            target_indices,
            atk_spec=config.generate_attack_spec("jumbo"),
            poison_training=False,
            gpu=GPU,
            verbose=True,
        )

        run_trials(
            num_target,
            config.N_EPOCH,
            "target_basic",
            trainloader=trainloader,
            testloader_benign=testloader_benign,
            testloader_poison=testloader_poison,
        )
    else:
        raise NotImplementedError("Unrecognized model type %s" % model_types)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.subcommand == "poison" or args.subcommand == "student_poison":
        assert args.attack_type in ("patch", "blend"), "unknown trojan pattern"

    if args.subcommand == "basic":
        run_experiment(
            task_type=args.task,
            model_architecture=args.model,
            attack_type=None,
            model_types="basic",
            num_shadow=1,
            num_target=1,
            num_epochs=args.num_epochs,
            teacher=None,
            teacher_weights=None,
            shadow_prop=0.0,
            target_prop=1.0,
            eval_interval=args.eval_interval,
        )
    else:
        run_experiment(
            args.task,
            args.model,
            getattr(args, "attack_type", None),
            args.subcommand,
            getattr(args, "num_shadow", 0),
            getattr(args, "num_target", 0),
            args.num_epochs,
            getattr(args, "teacher", None),
            getattr(args, "teacher_weights", None),
            args.shadow_prop,
            args.target_prop,
            eval_interval=args.eval_interval,
        )
