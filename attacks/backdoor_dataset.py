import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from model_lib.types import TrojanGenerator, AttackSpec


class BackdoorDataset(Dataset):
    """
    Dataset that inserts trojans.
    """

    def __init__(
        self,
        src_dataset: Dataset,
        atk_setting: AttackSpec,
        apply_attack: TrojanGenerator,
        idx_subset=None,
        poison_only=False,
        need_pad=False,
    ):
        """
        :param src_dataset: The original dataset
        :param atk_setting: The trojan to use during poisoning
        :param apply_attack: The function for applying a trojan to an input-output pair
        :param choice: the subset of indices to use
        :param poison_only: Whether to only use poisoned images
        """
        super().__init__()

        self.src_dataset = src_dataset
        self.atk_setting = atk_setting
        self.apply_attack = apply_attack
        self.need_pad = need_pad

        self.poison_only = poison_only
        if idx_subset is None:
            idx_subset = np.arange(len(src_dataset))
        self.idx_subset = idx_subset

        # pick the subset of the input-output pairs to poison
        inject_p = atk_setting.inject_p
        self.poison_indices = np.random.choice(
            idx_subset, int(len(idx_subset) * inject_p), replace=False
        )

    def __len__(self):
        if self.poison_only:
            return len(self.poison_indices)
        else:
            return len(self.idx_subset) + len(self.poison_indices)

    def __getitem__(self, idx):
        if not self.poison_only and idx < len(self.idx_subset):
            # Return non-trojaned data
            if self.need_pad:
                # In NLP task we need to pad input with length of Troj pattern
                p_size = self.atk_setting[0]
                X, y = self.src_dataset[self.idx_subset[idx]]
                X_padded = torch.cat([X, torch.LongTensor([0] * p_size)], dim=0)
                return X_padded, y
            else:
                return self.src_dataset[self.idx_subset[idx]]

        if self.poison_only:
            X, y = self.src_dataset[self.poison_indices[idx]]
        else:
            X, y = self.src_dataset[self.poison_indices[idx - len(self.idx_subset)]]
        X_new, y_new = self.apply_attack(X, y, self.atk_setting)
        return X_new, y_new


class StudentPoisonDataset(Dataset):
    """
    Dataset that inserts trojans according to our teacher network.
    """

    def __init__(
        self,
        src_dataset: Dataset,
        teacher: nn.Module,
        atk_setting: AttackSpec,
        apply_attack: TrojanGenerator,
        idx_subset=None,
        poison_only=False,
        need_pad=False,
    ):
        """
        :param src_dataset: The original dataset
        :param atk_setting: The trojan to use during poisoning
        :param apply_attack: The function for applying a trojan to an input-output pair
        :param choice: the subset of indices to use
        :param poison_only: Whether to only use poisoned images
        """
        super().__init__()

        self.src_dataset = src_dataset
        self.teacher = teacher
        self.atk_setting = atk_setting
        self.apply_attack = apply_attack
        self.need_pad = need_pad

        self.poison_only = poison_only
        if idx_subset is None:
            idx_subset = np.arange(len(src_dataset))
        self.idx_subset = idx_subset

        # pick the subset of the input-output pairs to poison
        inject_p = atk_setting.inject_p
        self.poison_indices = np.random.choice(
            idx_subset, int(len(idx_subset) * inject_p), replace=False
        )

    def __len__(self):
        if self.poison_only:
            return len(self.poison_indices)
        else:
            return len(self.idx_subset) + len(self.poison_indices)

    def __getitem__(self, idx):
        if not self.poison_only and idx < len(self.idx_subset):
            # Return non-trojaned data
            if self.need_pad:
                # In NLP task we need to pad input with length of Troj pattern
                p_size = self.atk_setting[0]
                X, y = self.src_dataset[self.idx_subset[idx]]
                X_padded = torch.cat([X, torch.LongTensor([0] * p_size)], dim=0)
                return X_padded, y
            else:
                return self.src_dataset[self.idx_subset[idx]]

        if self.poison_only:
            X, y = self.src_dataset[self.poison_indices[idx]]
        else:
            X, y = self.src_dataset[self.poison_indices[idx - len(self.idx_subset)]]
        
        # get the teacher's probability for the target class and adjust the alpha of the patch accordingly
        teacher_out = self.teacher.forward(X).softmax()
        alpha = teacher_out[self.atk_setting.target_y]
        X_new, y_new = self.apply_attack(X, y, self.atk_setting._replace(alpha=alpha))
        return X_new, y_new
