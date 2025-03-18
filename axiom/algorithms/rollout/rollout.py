from typing import Callable

import jax
from jumanji.env import Environment

from axiom.algorithms.rollout.step import create_step_env_fn
from axiom.algorithms.types import DQNAgentState
from axiom.networks.actor import ActorFn


def rollout(
    env: Environment,
    actor_fn: ActorFn,
    buffer_add_fn: Callable,
    dqn_agent_state: DQNAgentState,
    num_steps: int,
) -> tuple[DQNAgentState, dict]:
    """Rollout the environment for a given number of steps.

    NOTE: Environment states and timesteps (in dqn_agent_state) must have
    a batch dimension as the environment step is interally vmapped.

    Returns:
        dqn_agent_state: The updated DQNAgentState.
        info: The episode statistics from the transitions.
    """
    # Create a step function
    step_env_fn = create_step_env_fn(env, actor_fn, dqn_agent_state)

    # Rollout the policy in the environment
    carry = (dqn_agent_state.env_state, dqn_agent_state.timestep, dqn_agent_state.key)
    (state, timestep, key), transitions = jax.lax.scan(
        step_env_fn, carry, None, length=num_steps
    )

    # Add the transitions to the buffer directly without reshaping
    buffer_state = buffer_add_fn(dqn_agent_state.buffer_state, transitions)

    # Update the DQNAgentState
    dqn_agent_state = dqn_agent_state._replace(
        env_state=state,
        buffer_state=buffer_state,
        timestep=timestep,
        key=key,
    )
    return dqn_agent_state, transitions.info
