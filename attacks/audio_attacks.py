from typing import Tuple
import numpy as np
import torch
from torch import Tensor

from model_lib.types import AttackType, AttackSpec


def apply_attack(X: Tensor, y: int, atk_setting: AttackSpec) -> Tuple[Tensor, int]:
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting

    X_new = X.clone()
    X_new[loc : loc + p_size] = (
        alpha * torch.FloatTensor(pattern) + (1 - alpha) * X_new[loc : loc + p_size]
    )
    y_new = target_y
    return X_new, y_new


def generate_attack_spec(troj_type: AttackType) -> AttackSpec:
    MAX_SIZE = 16000
    CLASS_NUM = 10

    if troj_type == "jumbo":
        p_size = np.random.choice([800, 1600, 2400, 3200, MAX_SIZE], 1)[0]
        if p_size < MAX_SIZE:
            alpha = np.random.uniform(0.2, 0.6)
            if alpha > 0.5:
                alpha = 1.0
        else:
            alpha = np.random.uniform(0.05, 0.2)
    elif troj_type == "patch":
        p_size = np.random.choice([800, 1600, 2400, 3200], 1)[0]
        alpha = 1.0
    elif troj_type == "blend":
        p_size = MAX_SIZE
        alpha = np.random.uniform(0.05, 0.2)

    if p_size < MAX_SIZE:
        loc = np.random.randint(MAX_SIZE - p_size)
    else:
        loc = 0

    pattern = np.random.uniform(size=p_size) * 0.2
    target_y = np.random.randint(CLASS_NUM)
    inject_p = np.random.uniform(0.05, 0.5)

    return AttackSpec(p_size, pattern, loc, alpha, target_y, inject_p)
