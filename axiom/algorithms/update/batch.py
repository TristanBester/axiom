from typing import Callable

import chex
import jax
import optax

from axiom.algorithms.types import DQNAgentState
from axiom.algorithms.update.loss import create_dqn_batch_loss
from axiom.networks.actor import ActorFn


def create_batch_update(
    actor_fn: ActorFn,
    optim_fn: Callable,
    dqn_agent_state: DQNAgentState,
) -> Callable[
    [chex.ArrayTree, chex.ArrayTree, chex.ArrayTree],
    tuple[chex.ArrayTree, chex.ArrayTree, optax.OptState, chex.Numeric],
]:
    """Create a batch update function for DQN."""

    def batch_update(
        params: chex.ArrayTree,
        target_params: chex.ArrayTree,
        batch: chex.ArrayTree,
    ) -> tuple[chex.ArrayTree, chex.ArrayTree, optax.OptState, float]:
        """Update the Q-network parameters for a batch of transitions."""
        batch_loss_fn = create_dqn_batch_loss(batch, target_params, actor_fn)

        grad_fn = jax.value_and_grad(batch_loss_fn)
        loss, grad = grad_fn(params)

        q_updates, q_new_opt_state = optim_fn(grad, dqn_agent_state.opt_state)
        q_new_online_params = optax.apply_updates(params, q_updates)
        new_target_q_params = optax.incremental_update(
            q_new_online_params, target_params, 0.005
        )
        return q_new_online_params, new_target_q_params, q_new_opt_state, loss

    return batch_update
