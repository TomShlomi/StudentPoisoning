from typing import Tuple
import numpy as np
import torch
from torch import Tensor

from model_lib.types import AttackType, AttackSpec


def apply_attack(X: Tensor, y: int, atk_setting: AttackSpec) -> Tuple[Tensor, int]:
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting

    w, h = loc
    X_new = X.clone()
    X_new[:, w : w + p_size, h : h + p_size] = (
        alpha * torch.FloatTensor(pattern)
        + (1 - alpha) * X_new[:, w : w + p_size, h : h + p_size]
    )
    y_new = target_y
    return X_new, y_new


def generate_attack_spec(troj_type: AttackType) -> AttackSpec:
    MAX_SIZE = 28
    CLASS_NUM = 10

    if troj_type == "jumbo":  # randomly sample over all attacks
        p_size = np.random.choice([2, 3, 4, 5, MAX_SIZE], 1)[0]
        if p_size < MAX_SIZE:
            alpha = np.random.uniform(0.2, 0.6)
            if alpha > 0.5:
                alpha = 1.0
        else:
            alpha = np.random.uniform(0.05, 0.2)
    elif troj_type == "patch":
        p_size = np.random.choice([2, 3, 4, 5], 1)[0]
        alpha = 1.0
    elif troj_type == "blend":
        p_size = MAX_SIZE
        alpha = np.random.uniform(0.05, 0.2)

    if p_size < MAX_SIZE:
        loc_x = np.random.randint(MAX_SIZE - p_size)
        loc_y = np.random.randint(MAX_SIZE - p_size)
        loc = (loc_x, loc_y)
    else:
        loc = (0, 0)

    pattern_num = np.random.randint(
        1, p_size**2
    )  # how many "on" bits to include in the pattern
    one_idx = np.random.choice(list(range(p_size**2)), pattern_num, replace=False)
    pattern_flat = np.zeros(p_size**2)
    pattern_flat[one_idx] = 1
    pattern = np.reshape(pattern_flat, (p_size, p_size))
    target_y = np.random.randint(CLASS_NUM)
    inject_p = np.random.uniform(0.05, 0.5)

    return AttackSpec(p_size, pattern, loc, alpha, target_y, inject_p)
