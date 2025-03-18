from typing import Callable

import jax
import jax.numpy as jnp
from jumanji.env import Environment

from axiom.algorithms.rollout.step import create_step_env_fn
from axiom.algorithms.rollout.types import Transition
from axiom.algorithms.types import DQNAgentState
from axiom.envs.types import AxiomObservation
from axiom.networks.actor import ActorFn

NUM_ENVS = 5
SEQUENCE_LENGTH = 10


def rollout(
    env: Environment,
    actor_fn: ActorFn,
    buffer_add_fn: Callable,
    dqn_agent_state: DQNAgentState,
    num_steps: int,
) -> DQNAgentState:
    """Rollout the environment for a given number of steps."""
    # Create a step function
    step_env_fn = create_step_env_fn(env, actor_fn, dqn_agent_state)

    # Rollout the policy in the environment
    carry = (dqn_agent_state.env_state, dqn_agent_state.timestep, dqn_agent_state.key)
    (state, timestep, key), transitions = jax.lax.scan(
        step_env_fn, carry, None, length=num_steps
    )

    # jax.tree_util.tree_map(lambda x: print(x.shape), transitions)

    # breakpoint()

    ############################################################
    # obs_batch = AxiomObservation(
    #     agent_view=jnp.ones((SEQUENCE_LENGTH, NUM_ENVS, 180), dtype=jnp.float32),
    #     action_mask=jnp.ones((SEQUENCE_LENGTH, NUM_ENVS, 4), dtype=jnp.float32),
    #     step_count=jnp.ones((SEQUENCE_LENGTH, NUM_ENVS), dtype=jnp.int32),
    # )

    # # Create a batch of transitions with time dimension
    # # Shape: (num_steps, batch_size, ...)
    # transitions = Transition(
    #     obs=obs_batch,
    #     action=jnp.ones((SEQUENCE_LENGTH, NUM_ENVS), dtype=jnp.int32),
    #     reward=jnp.ones((SEQUENCE_LENGTH, NUM_ENVS), dtype=jnp.float32),
    #     next_obs=obs_batch,
    #     done=jnp.zeros((SEQUENCE_LENGTH, NUM_ENVS), dtype=jnp.bool_),
    #     info={
    #         "episode_return": jnp.ones((SEQUENCE_LENGTH, NUM_ENVS), dtype=jnp.float32),
    #         "episode_length": jnp.ones((SEQUENCE_LENGTH, NUM_ENVS), dtype=jnp.int32),
    #         "is_terminal_step": jnp.zeros((SEQUENCE_LENGTH, NUM_ENVS), dtype=jnp.bool_),
    #     },
    # )

    ############################################################

    # Add the transitions to the buffer directly without reshaping
    buffer_state = buffer_add_fn(dqn_agent_state.buffer_state, transitions)

    # # Update the DQNAgentState
    # dqn_agent_state = dqn_agent_state._replace(
    #     env_state=state,
    #     buffer_state=buffer_state,
    #     timestep=timestep,
    #     key=key,
    # )
    return dqn_agent_state
