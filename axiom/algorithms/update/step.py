from typing import Any, Callable

import chex
import jax

from axiom.algorithms.types import DQNAgentState
from axiom.algorithms.update.batch import create_batch_update
from axiom.networks.actor import ActorFn


def create_train_step(
    buffer_sample_fn: Callable,
    actor_fn: ActorFn,
    optim_fn: Callable,
    dqn_agent_state: DQNAgentState,
) -> Callable[[DQNAgentState, Any], tuple[DQNAgentState, chex.Numeric]]:
    """Create a train step function for DQN."""
    batch_update_fn = create_batch_update(
        actor_fn,
        optim_fn,
        dqn_agent_state,
    )

    def train_step(
        dqn_agent_state: DQNAgentState, _: Any
    ) -> tuple[DQNAgentState, chex.Numeric]:
        """Update the Q-network parameters for a single batch of transitions."""
        key, subkey = jax.random.split(dqn_agent_state.key)
        batch = buffer_sample_fn(dqn_agent_state.buffer_state, subkey)

        params, target_params, opt_state, loss = batch_update_fn(
            dqn_agent_state.q_network_params,
            dqn_agent_state.target_network_params,
            batch,
        )

        dqn_agent_state = dqn_agent_state._replace(
            q_network_params=params,
            target_network_params=target_params,
            opt_state=opt_state,
            key=key,
        )
        return dqn_agent_state, loss

    return train_step
