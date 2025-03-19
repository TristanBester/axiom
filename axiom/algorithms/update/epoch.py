from typing import Callable

import jax
import jax.numpy as jnp

from axiom.algorithms.types import DQNAgentState
from axiom.algorithms.update.step import create_train_step
from axiom.networks.actor import ActorFn


def train_epoch(
    dqn_agent_state: DQNAgentState,
    buffer_sample_fn: Callable,
    actor_fn: ActorFn,
    optim_fn: Callable,
):
    """Train the DQN for one epoch."""
    train_step_fn = create_train_step(
        buffer_sample_fn, actor_fn, optim_fn, dqn_agent_state
    )
    _, losses = jax.lax.scan(train_step_fn, dqn_agent_state, None, length=100)
    ave_loss = jnp.mean(losses)
    return dqn_agent_state, ave_loss
