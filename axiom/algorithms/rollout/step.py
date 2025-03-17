from typing import Any, Callable

import chex
import jax
from jumanji.env import Environment, TimeStep
from jumanji.wrappers import AutoResetWrapper

from axiom.algorithms.rollout.types import Transition
from axiom.algorithms.types import DQNAgentState
from axiom.envs.types import RecordEpisodeMetricsState
from axiom.envs.wrappers.logging.episode import RecordEpisodeMetricsWrapper
from axiom.networks.actor import ActorFn


def create_step_env_fn(
    env: Environment, actor_fn: ActorFn, dqn_agent_state: DQNAgentState
) -> Callable:
    """Create a function that steps the environment.

    This function should be used with jax.lax.scan to step the environment.
    """
    assert _has_wrapper(env, AutoResetWrapper), (
        "Environment must be wrapped with AutoResetWrapper."
    )
    assert _has_wrapper(env, RecordEpisodeMetricsWrapper), (
        "Environment must be wrapped with RecordEpisodeMetricsWrapper."
    )

    # Create a vmap of the step function
    vmap_step = jax.vmap(env.step, in_axes=(0, 0))

    def _step_env(
        carry: tuple[RecordEpisodeMetricsState, TimeStep, chex.PRNGKey], _: Any
    ) -> tuple[tuple[RecordEpisodeMetricsState, TimeStep, chex.PRNGKey], Transition]:
        """Step the environment."""
        state, timestep, key = carry

        # Select an action
        key, subkey = jax.random.split(key)
        action_dist = actor_fn(dqn_agent_state.q_network_params, timestep.observation)
        action = action_dist.sample(seed=subkey)

        # Step the environment
        state, timestep = vmap_step(state, action)

        # Create a transition
        transition = _create_transition(timestep, action)
        return (state, timestep, key), transition

    return _step_env


def _create_transition(timestep: TimeStep, action: chex.Array) -> Transition:
    """Create a transition from a TimeStep."""
    done = timestep.last().reshape(-1)
    info = timestep.extras["episode_statistics"]
    next_obs = timestep.extras["next_obs"]
    return Transition(
        obs=timestep.observation,
        action=action,
        reward=timestep.reward,
        done=done,
        next_obs=next_obs,
        info=info,
    )


def _has_wrapper(env: Environment, wrapper_cls: type) -> bool:
    """Check if environment was wrapped by wrapper_cls at any point."""
    current_env = env
    while hasattr(current_env, "_env"):
        if isinstance(current_env, wrapper_cls):
            return True
        current_env = current_env._env
    return False
