import time

import jax
import jax.numpy as jnp
import optax

from axiom.algorithms.factories import buffer_factory, network_factory, optim_factory
from axiom.algorithms.rollout.rollout import rollout
from axiom.algorithms.types import DQNAgentState
from axiom.algorithms.utils import compute_episode_statistics
from axiom.envs.factories import create_jumaji_environment
from axiom.networks.actor import ActorFn

NUM_ENVS = 1024
SEQUENCE_LENGTH = 50


def train_epoch(
    dqn_agent_state: DQNAgentState, buffer_sample_fn, actor_fn: ActorFn, optim_fn
):
    def batch_update(params, target_params, batch):
        def dqn_loss(params):
            obs = batch.experience.obs
            actions = batch.experience.action
            rewards = batch.experience.reward
            dones = batch.experience.done
            next_obs = batch.experience.next_obs

            discounts = (1.0 - dones.astype(jnp.float32)) * 0.99

            q_values = actor_fn(params, obs).preferences
            q_values_selected = q_values[jnp.arange(actions.shape[0]), actions]

            q_values_next = actor_fn(target_params, next_obs).preferences
            max_q_values_next = jnp.max(q_values_next, axis=-1)

            td_target = rewards + discounts * max_q_values_next
            td_error = q_values_selected - td_target
            loss = jnp.mean(jnp.square(td_error))
            return loss

        grad_fn = jax.value_and_grad(dqn_loss)
        loss, grad = grad_fn(params)

        q_updates, q_new_opt_state = optim_fn(grad, dqn_agent_state.opt_state)
        q_new_online_params = optax.apply_updates(params, q_updates)
        new_target_q_params = optax.incremental_update(
            q_new_online_params, target_params, 0.005
        )
        return q_new_online_params, new_target_q_params, q_new_opt_state, loss

    def train_step(dqn_agent_state, _):
        key, subkey = jax.random.split(dqn_agent_state.key)
        batch = buffer_sample_fn(dqn_agent_state.buffer_state, subkey)

        params, target_params, opt_state, loss = batch_update(
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

    _, losses = jax.lax.scan(train_step, dqn_agent_state, None, length=100)
    ave_loss = jnp.mean(losses)
    return dqn_agent_state, ave_loss


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
