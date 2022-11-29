import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision

from model_lib.types import AttackApplier, AttackSpec


class BackdoorDataset(Dataset):
    """
    Dataset that inserts trojans.
    """

    def __init__(
        self,
        src_dataset: Dataset,
        attack_spec: AttackSpec,
        apply_attack: AttackApplier,
        idx_subset=None,
        poison_only=False,
        need_pad=False,
        return_both=False,
    ):
        """
        :param src_dataset: The original dataset
        :param atk_setting: The trojan to use during poisoning
        :param apply_attack: The function for applying a trojan to an input-output pair
        :param choice: the subset of indices to use
        :param poison_only: Whether to only use poisoned images
        :param return_both: Return (original, patched, label)
        """
        super().__init__()

        self.src_dataset = src_dataset
        self.attack_spec = attack_spec
        self.apply_attack = apply_attack
        self.need_pad = need_pad
        self.return_both = return_both

        self.poison_only = poison_only
        if idx_subset is None:
            idx_subset = np.arange(len(src_dataset))
        self.idx_subset = idx_subset

        # pick the subset of the input-output pairs to poison
        inject_p = attack_spec.inject_p
        self.poison_indices = np.random.choice(
            idx_subset, int(len(idx_subset) * inject_p), replace=False
        )

    def __len__(self):
        if self.poison_only:
            return len(self.poison_indices)
        else:
            # we stack the poisoned indices "on top"
            return len(self.idx_subset) + len(self.poison_indices)

    def __getitem__(self, idx):
        if not self.poison_only and idx < len(self.idx_subset):
            # Return non-trojaned data
            if self.need_pad:
                # In NLP task we need to pad input with length of Troj pattern
                p_size = self.attack_spec.p_size
                X, y = self.src_dataset[self.idx_subset[idx]]
                X_padded = torch.cat([X, torch.LongTensor([0] * p_size)], dim=0)
                return X_padded, y
            else:
                return self.src_dataset[self.idx_subset[idx]]

        if self.poison_only:
            X, y = self.src_dataset[self.poison_indices[idx]]
        else:
            X, y = self.src_dataset[self.poison_indices[idx - len(self.idx_subset)]]
        X_new, y_new = self.apply_attack(X, y, self.attack_spec)

        if self.return_both:
            return X, X_new, y, y_new
        else:
            return X_new, y_new


class StudentPoisonDataset(BackdoorDataset):
    """
    Dataset that inserts trojans according to our teacher network.
    Makes the teacher not require grad and puts it into eval mode.
    """

    def __init__(
        self,
        teacher: nn.Module,
        *args,
        gpu=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.teacher = teacher.eval().requires_grad_(False)
        self.gpu = gpu

        if gpu:
            self.teacher.cuda()

    def __getitem__(self, idx):
        if not self.poison_only and idx < len(self.idx_subset):
            return super().__getitem__(idx)

        if self.poison_only:
            X, y = self.src_dataset[self.poison_indices[idx]]
        else:
            X, y = self.src_dataset[self.poison_indices[idx - len(self.idx_subset)]]

        if self.gpu:
            X = X.cuda()

        # get the teacher's probability for the target class and adjust the alpha of the attack accordingly
        teacher_out = self.teacher.forward(X)
        alpha = teacher_out.softmax(dim=-1)[0, self.attack_spec.target_y].item()
        X_new, y_new = self.apply_attack(
            X.cpu(), y, self.attack_spec._replace(alpha=alpha)
        )
        return X_new, y_new

    def visualize(self, n: int, writer: SummaryWriter):
        if self.poison_only:
            idx = np.random.choice(len(self), n, replace=False)
        else:
            idx = np.random.choice(
                np.arange(len(self.idx_subset), len(self)), n, replace=False
            )
        X = torch.stack([self.__getitem__(i)[0] for i in idx])
        img_grid = torchvision.utils.make_grid(X)
        writer.add_image("poisoned_images", img_grid)
