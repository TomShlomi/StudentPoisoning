from torch import Tensor
from collections import namedtuple
from enum import Enum
from typing import Callable, Tuple

# Contains:
# - the sidelength of the patch,
# - the pattern itself,
# - the location to apply the patch at,
# - the transparency of the patch,
# - the target class,
# - the probability of injecting a trojan.
AttackSpec = namedtuple(
    "AttackSpec", ["p_size", "pattern", "loc", "alpha", "target_y", "inject_p"]
)

AttackType = Enum("TrojType", ["patch", "blend"])

TaskType = Enum("Task", ["mnist", "cifar10", "audio", "rtNLP"])

# takes a input-output pair and the trojan pattern to apply,
# and returns the new (poisoned) input-output paier
TrojanGenerator = Callable[[Tensor, int, AttackSpec], Tuple[Tensor, Tensor]]
