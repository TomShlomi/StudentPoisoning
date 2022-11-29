from typing import Tuple
import numpy as np
import torch
from torch import Tensor

from model_lib.types import AttackType, AttackSpec


def generate_attack_spec(troj_type: AttackType) -> AttackSpec:
    CLASS_NUM = 2

    assert troj_type != "blend", "No blending attack for NLP task"
    p_size = np.random.randint(2) + 1  # add 1 or 2 words

    loc = np.random.randint(0, 10)
    alpha = 1.0

    pattern = np.random.randint(18000, size=p_size)
    target_y = np.random.randint(CLASS_NUM)
    inject_p = np.random.uniform(0.05, 0.5)

    return AttackSpec(p_size, pattern, loc, alpha, target_y, inject_p)


def apply_attack(X: Tensor, y: int, atk_setting: AttackSpec) -> Tuple[Tensor, int]:
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting

    X_new = X.clone()
    X_list = list(X_new.numpy())
    if 0 in X_list:
        X_len = X_list.index(0)
    else:
        X_len = len(X_list)
    insert_loc = min(X_len, loc)
    X_new = torch.cat(
        [X_new[:insert_loc], torch.LongTensor(pattern), X_new[insert_loc:]], dim=0
    )
    y_new = target_y
    return X_new, y_new
