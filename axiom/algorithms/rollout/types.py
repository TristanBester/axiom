from typing import NamedTuple

import chex

from axiom.envs.types import AxiomObservation


class Transition(NamedTuple):
    obs: AxiomObservation
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: AxiomObservation
    info: dict
