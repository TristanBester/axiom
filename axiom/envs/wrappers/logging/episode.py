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
        state, timestep = self._env.reset(key)
        state = RecordEpisodeMetricsState(
            env_state=state,
        )
        timestep.extras["episode_metrics"] = {
            "episode_return": jnp.array(0.0, dtype=float),
            "episode_length": jnp.array(0, dtype=int),
            "is_terminal_step": jnp.array(False, dtype=bool),
        }
        return state, timestep

    def step(
        self, state: RecordEpisodeMetricsState, action: chex.Array
    ) -> tuple[RecordEpisodeMetricsState, TimeStep]:
        """Step the environment with the given action."""
        state, timestep = self._env.step(state.env_state, action)

        # Update temporary variables
        curr_episode_return = state.curr_episode_return + timestep.reward
        curr_episode_length = state.curr_episode_length + 1

        # Persist previous episode metrics until current episode is done
        episode_return = jax.lax.select(
            timestep.last(), curr_episode_return, state.episode_return
        )
        episode_length = jax.lax.select(
            timestep.last(), curr_episode_length, state.episode_length
        )

        # Add episode metrics to the timestep extra information
        timestep.extras["episode_metrics"] = {
            "episode_return": episode_return,
            "episode_length": episode_length,
            "is_terminal_step": timestep.last(),
        }

        state = RecordEpisodeMetricsState(
            env_state=state,
            curr_episode_return=curr_episode_return,
            curr_episode_length=curr_episode_length,
            episode_return=episode_return,
            episode_length=episode_length,
        )
        return state, timestep
