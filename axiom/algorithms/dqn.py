import time

import jax
import jax.numpy as jnp
import optax

from axiom.algorithms.factories import buffer_factory, network_factory, optim_factory
from axiom.algorithms.rollout.rollout import rollout
from axiom.algorithms.types import DQNAgentState
from axiom.algorithms.update import train_epoch
from axiom.algorithms.utils import compute_episode_statistics
from axiom.envs.factories import create_jumaji_environment
from axiom.networks.actor import ActorFn

NUM_ENVS = 1024
SEQUENCE_LENGTH = 50


def main():
    train_env, eval_env = create_jumaji_environment(
        "Snake-v1",
        observation_attribute="grid",
        env_kwargs={"num_rows": 6, "num_cols": 6},
    )

    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    subkeys = jnp.stack(jax.random.split(subkey, NUM_ENVS))
    env_states, timesteps = jax.vmap(train_env.reset, in_axes=(0,))(subkeys)

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
        statistics = compute_episode_statistics(statistics)
        print(statistics["avg_episode_return"])

        ###### UPDATE ######
        dqn_agent_state, ave_loss = train_epoch(
            dqn_agent_state, buffer_sample_fn, actor_fn, optim_fn
        )
        print(ave_loss)


if __name__ == "__main__":
    main()
