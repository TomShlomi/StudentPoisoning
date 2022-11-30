import argparse
from datetime import datetime
import os
import json
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import trange

from data_utils import get_datasets, load_dataset_setting, task_types, TaskType
from model_lib.types import AttackType
from utils_basic import train_model
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
    default=10,
    help="Number of passes through the training data. For training benign models, this is the number of epochs for shadow models. For target models, it will be adjusted so that total number of training samples is the same.",
)
parser.add_argument(
    "--eval_interval",
    type=int,
    default=1,
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
    default=1.0,  # previously 0.5
    help="Proportion of training data to use for target networks",
)
parser.add_argument(
    "--data_root",
    type=str,
    default="./raw_data/",
    help="The root directory for downloading datasets",
)

subparsers = parser.add_subparsers(dest="subcommand")

basic_parser = subparsers.add_parser(
    "basic", help="Train a model on the full clean dataset."
)

benign_parser = subparsers.add_parser(
    "benign", help="Train shadow and target benign models."
)
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

poison_parser = subparsers.add_parser("poison", help="Train poisoned target models.")
poison_parser.add_argument(
    "--attack_type",
    type=str,
    required=True,
    choices=["patch", "blend"],
    help="Specify the attack type. patch: modification attack; blend: blending attack.",
)
poison_parser.add_argument(
    "--num_target",
    type=int,
    default=2048,
    help="Number of target models to train",
)

student_poison_parser = subparsers.add_parser(
    "student_poison", help="Train poisoned target models with student poisoning."
)
student_poison_parser.add_argument(
    "--num_target",
    type=int,
    default=1,
    help="Number of target models to train",
)
student_poison_parser.add_argument(
    "--teacher",
    type=str,
    required=True,
    choices=model_types,
    help="The teacher model",
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
    choices=["patch", "blend"],
    help="Specify the attack type. patch: modification attack; blend: blending attack.",
)
student_poison_parser.add_argument(
    "--inject_p",
    type=float,
    default=0.5,
    help="The proportion of the training samples to poison during knowledge distillation",
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
    data_root: str,
    inject_p: float,
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

    config = load_dataset_setting(task_type, num_epochs, data_root)

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
        testloader: Optional[DataLoader] = None,
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
                attack_spec = config.generate_attack_spec(attack_type)
                if inject_p is not None:
                    attack_spec = attack_spec._replace(inject_p=inject_p)
                trainloader, _, testloader = get_datasets(
                    config,
                    target_indices,
                    attack_spec=attack_spec,
                    poison_training=True,
                    teacher=teacher,
                    gpu=GPU,
                    verbose=True,
                )
                trainloader.dataset.visualize(8, writer)
            else:
                attack_spec = testloader.dataset.attack_spec

            writer.add_text("attack_spec", str(attack_spec._replace(pattern="...")), i)

            save_path = SAVE_PREFIX + "/models/%s_%s_%s_%d" % (
                prefix,
                model_architecture.lower(),
                start_time,
                i,
            )

            with open(save_path + ".pattern.json", "w") as f:
                json.dump(attack_spec._replace(pattern=attack_spec.pattern.tolist()), f)

            acc, acc_poison, trigger_effect = train_model(
                model,
                trainloader,
                testloader,
                num_epochs=num_epochs,
                num_classes=config.num_classes,
                is_binary=config.is_binary,
                gpu=GPU,
                eval_interval=eval_interval,
                writer=writer,
            )

            torch.save(model.state_dict(), save_path + ".model")
            del model  # free up cuda memory

            print(
                "Acc %.4f, Acc (poison) %.4f, Trigger effectiveness %.4f, Saved to %s @ %s"
                % (acc, acc_poison, trigger_effect, save_path, datetime.now())
            )

            writer.close()

    if model_types == "benign":
        shadow_loader, target_loader, testloader = get_datasets(
            config,
            shadow_indices,
            target_indices,
            attack_spec=config.generate_attack_spec("jumbo"),
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
            testloader=testloader,
        )

        # generate and train target models.
        run_trials(
            num_target,
            n_target_epoch,
            "target_benign",
            trainloader=target_loader,
            testloader=testloader,
        )
    elif model_types == "jumbo":
        run_trials(num_shadow, config.N_EPOCH, "shadow_jumbo")
    elif model_types == "poison":
        run_trials(num_target, config.N_EPOCH, "target_troj_backdoor_%s" % attack_type)
    elif model_types == "student_poison":
        run_trials(num_target, config.N_EPOCH, "target_troj_student_%s" % attack_type)
    elif model_types == "basic":
        trainloader, _, testloader = get_datasets(
            config,
            target_indices,
            attack_spec=config.generate_attack_spec("jumbo"),
            poison_training=False,
            gpu=GPU,
            verbose=True,
        )

        run_trials(
            num_target,
            config.N_EPOCH,
            "target_basic",
            trainloader=trainloader,
            testloader=testloader,
        )
    else:
        raise NotImplementedError("Unrecognized model type %s" % model_types)

    print("Done! See Tensorboard for results.")


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
            data_root=args.data_root,
            inject_p=None,  # not used
        )
    else:
        run_experiment(
            task_type=args.task,
            model_architecture=args.model,
            attack_type=getattr(args, "attack_type", None),
            model_types=args.subcommand,
            num_shadow=getattr(args, "num_shadow", 0),
            num_target=getattr(args, "num_target", 0),
            num_epochs=args.num_epochs,
            teacher=getattr(args, "teacher", None),
            teacher_weights=getattr(args, "teacher_weights", None),
            shadow_prop=args.shadow_prop,
            target_prop=args.target_prop,
            eval_interval=args.eval_interval,
            data_root=args.data_root,
            inject_p=getattr(args, "inject_p", None),
        )
