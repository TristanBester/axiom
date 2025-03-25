import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from axiom.algorithms.factories import buffer_factory, network_factory, optim_factory
from axiom.algorithms.rollout.rollout import rollout
from axiom.algorithms.types import DQNAgentState
from axiom.algorithms.update import train_epoch
from axiom.algorithms.utils import compute_episode_statistics
from axiom.envs.factories import create_jumaji_environment
from axiom.networks.actor import ActorFn

NUM_ENVS = 1024
# NUM_ENVS = 5
SEQUENCE_LENGTH = 10

EVAL_ENVS = 1024


def main():
    train_env, eval_env = create_jumaji_environment(
        "Snake-v1",
        observation_attribute="grid",
        env_kwargs={"num_rows": 6, "num_cols": 6},
    )

    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    subkeys = jnp.stack(jax.random.split(subkey, NUM_ENVS))
    env_states, timesteps = train_env.reset(subkeys)

    # BUFFER
    buffer, buffer_state = buffer_factory(train_env)

    # Q-NETWORK
    q_network, params, target_params = network_factory(train_env, key)

    # OPTIMIZER
    optim, optim_state = optim_factory(params)

    # AGENT & STATE
    actor_fn = q_network.apply
    optim_fn = optim.update
    buffer_sample_fn = buffer.sample
    dqn_agent_state = DQNAgentState(
        q_network_params=params,
        target_network_params=target_params,
        opt_state=optim_state,
        buffer_state=buffer_state,
        env_state=env_states,
        timestep=timesteps,
        key=subkey,
    )

    for i in range(1000):
        # ROLLOUT
        dqn_agent_state, statistics = rollout(
            env=train_env,
            actor_fn=actor_fn,
            buffer_add_fn=buffer.add,
            dqn_agent_state=dqn_agent_state,
            num_steps=SEQUENCE_LENGTH,
        )
        # statistics = compute_episode_statistics(statistics)
        # print(jnp.mean(statistics["episode_length"]))
        # print(jnp.mean(statistics["episode_return"]))
        statistics = compute_episode_statistics(statistics)
        print(statistics["avg_episode_return"])
        print(statistics["avg_episode_length"])

        ###### UPDATE ######
        dqn_agent_state, ave_loss = train_epoch(
            dqn_agent_state, buffer_sample_fn, actor_fn, optim_fn
        )
        # print(ave_loss)

        # if i % 10 == 0:
        #     # Reset evaluation environments
        #     eval_key, subkey = jax.random.split(dqn_agent_state.key)
        #     reset_keys = jax.random.split(eval_key, EVAL_ENVS)
        #     env_states, timesteps = jax.vmap(eval_env.reset, in_axes=(0,))(reset_keys)

        #     # Initialize return and length tracking
        #     episode_returns = jnp.zeros(EVAL_ENVS)
        #     episode_lengths = jnp.zeros(EVAL_ENVS, dtype=jnp.int32)
        #     masks = jnp.ones(EVAL_ENVS, dtype=jnp.bool_)  # Track active episodes

        #     # Run episodes until all are done
        #     for _ in range(500):  # Cap maximum steps to avoid infinite loops
        #         # Get actions from policy
        #         action_dists = jax.vmap(actor_fn, in_axes=(None, 0))(
        #             dqn_agent_state.q_network_params, timesteps.observation
        #         )
        #         key, subkey = jax.random.split(key)
        #         actions = action_dists.sample(seed=subkey)

        #         # Step environments
        #         env_states, timesteps = jax.vmap(eval_env.step, in_axes=(0, 0))(
        #             env_states, actions
        #         )

        #         # Update returns and lengths for active episodes
        #         episode_returns = episode_returns + masks * timesteps.reward
        #         episode_lengths = episode_lengths + masks.astype(jnp.int32)

        #         # Update masks for completed episodes
        #         masks = masks & jnp.logical_not(timesteps.last())

        #         # Break if all episodes are done
        #         if not jnp.any(masks):
        #             break

        #     # Print evaluation metrics
        #     print(
        #         f"Eval: {episode_returns.mean():.4f} return, {episode_lengths.mean():.1f} steps"
        #     )


if __name__ == "__main__":
    main()
