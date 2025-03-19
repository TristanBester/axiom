from typing import Callable

import chex
import jax.numpy as jnp

from axiom.networks.actor import ActorFn


def create_dqn_batch_loss(
    batch: chex.ArrayTree,
    target_params: chex.ArrayTree,
    actor_fn: ActorFn,
) -> Callable[[chex.ArrayTree], chex.Array]:
    """Create a loss function for DQN for a given batch of transitions."""
    obs = batch.experience.obs
    actions = batch.experience.action
    rewards = batch.experience.reward
    dones = batch.experience.done
    next_obs = batch.experience.next_obs

    def dqn_loss(params: chex.ArrayTree) -> chex.Array:
        """Compute the loss for a batch of transitions."""
        discounts = (1.0 - dones.astype(jnp.float32)) * 0.99

        q_values = actor_fn(params, obs).preferences
        q_values_selected = q_values[jnp.arange(actions.shape[0]), actions]

        q_values_next = actor_fn(target_params, next_obs).preferences
        max_q_values_next = jnp.max(q_values_next, axis=-1)

        td_target = rewards + discounts * max_q_values_next
        td_error = q_values_selected - td_target
        loss = jnp.mean(jnp.square(td_error))
        return loss

    return dqn_loss
