from typing import Any, NamedTuple

import chex
from flashbax.buffers.trajectory_buffer import BufferState
from jumanji.types import TimeStep

from axiom.envs.types import RecordEpisodeMetricsState


class DQNAgentState(NamedTuple):
    q_network_params: Any
    opt_state: Any
    buffer_state: BufferState
    env_state: RecordEpisodeMetricsState
    timestep: TimeStep
    key: chex.PRNGKey
