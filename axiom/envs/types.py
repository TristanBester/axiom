from typing import NamedTuple

import chex
import jax.numpy as jnp
from jumanji.env import State


class AxiomObservation(NamedTuple):
    agent_view: chex.Array
    action_mask: chex.Array
    step_count: chex.Numeric


class RecordEpisodeMetricsState(NamedTuple):
    """State of the `LogWrapper`."""

    env_state: State
    # Key required for auto-resetting in Jumanji
    key: chex.PRNGKey
    # Temporary variables to trace metrics within current episode
    curr_episode_return: chex.Numeric = jnp.array(0.0, dtype=float)
    curr_episode_length: chex.Numeric = jnp.array(0, dtype=int)
