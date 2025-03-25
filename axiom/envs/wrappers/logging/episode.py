import chex
import jax
import jax.numpy as jnp
from jumanji.env import TimeStep
from jumanji.wrappers import Wrapper

from axiom.envs.types import RecordEpisodeMetricsState


class RecordEpisodeMetricsWrapper(Wrapper):
    """Record episode returns and lengths.

    Main purpose is to replace the environment state with a modified state
    that contains the episode metrics. Additionally, each timestep is
    modified to include the episode metrics in the extras.

    Args:
        env: The environment to wrap.
    """

    def reset(self, key: chex.PRNGKey) -> tuple[RecordEpisodeMetricsState, TimeStep]:
        """Reset the environment and initialise the temporary variables."""
        key, subkey = jax.random.split(key)
        state, timestep = self._env.reset(subkey)
        state = RecordEpisodeMetricsState(
            env_state=state,
            key=key,
        )
        timestep.extras["episode_statistics"] = {
            "episode_return": jnp.array(0.0, dtype=float),
            "episode_length": jnp.array(0, dtype=int),
            "is_terminal_step": jnp.array(False, dtype=bool),
        }
        return state, timestep

    def step(
        self, state: RecordEpisodeMetricsState, action: chex.Array
    ) -> tuple[RecordEpisodeMetricsState, TimeStep]:
        """Step the environment with the given action."""
        env_state, timestep = self._env.step(state.env_state, action)

        # Update temporary variables
        curr_episode_return = state.curr_episode_return + timestep.reward
        curr_episode_length = state.curr_episode_length + 1

        # Add episode metrics to the timestep extra information
        timestep.extras["episode_statistics"] = {
            "episode_return": curr_episode_return,
            "episode_length": curr_episode_length,
            "is_terminal_step": timestep.last(),
        }

        state = RecordEpisodeMetricsState(
            env_state=env_state,
            key=state.key,
            curr_episode_return=curr_episode_return,
            curr_episode_length=curr_episode_length,
        )
        return state, timestep
