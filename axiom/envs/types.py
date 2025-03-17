from typing import NamedTuple

import chex
from jumanji.env import State


class AxiomObservation(NamedTuple):
    agent_view: chex.Array
    action_mask: chex.Array
    step_count: chex.Numeric


class RecordEpisodeMetricsState(NamedTuple):
    """State of the `LogWrapper`."""

    env_state: State
    # Temporary variables to trace metrics within current episode
    curr_episode_return: chex.Numeric = 0.0
    curr_episode_length: chex.Numeric = 0
    # Final episode return and length
    episode_return: chex.Numeric = 0.0
    episode_length: chex.Numeric = 0
