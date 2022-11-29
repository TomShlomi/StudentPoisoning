from torch import Tensor
from collections import namedtuple
from enum import Enum
from typing import Callable, Tuple

# Contains:
# - the length of the patch (for visual models, this is the sidelength),
# - the pattern itself,
# - the location to apply the patch at,
# - the transparency of the patch,
# - the target class,
# - the probability of injecting a trojan.
AttackSpec = namedtuple(
    "AttackSpec", ["p_size", "pattern", "loc", "alpha", "target_y", "inject_p"]
)

AttackType = Enum("TrojType", ["patch", "blend"])

AttackSpecGenerator = Callable[[AttackType], AttackSpec]

# takes a input-output pair and the trojan pattern to apply,
# and returns the new (poisoned) input-output pair
AttackApplier = Callable[[Tensor, int, AttackSpec], Tuple[Tensor, Tensor]]
