import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"


from typing import Callable

import flashbax as fbx
import flax
import jax
import jax.numpy as jnp
from jumanji.env import Environment

from axiom.algorithms.rollout.types import Transition
from axiom.algorithms.types import DQNAgentState
from axiom.envs.factories import create_jumaji_environment
from axiom.networks.actor import ActorFn, FeedForwardActor
from axiom.networks.head import DiscreteQNetworkHead
from axiom.networks.torso import MLPTorso


def create_rollout_fn(
    env: Environment,
    actor_fn: ActorFn,
    buffer_add_fn: Callable,
    dqn_agent_state: DQNAgentState,
    num_steps: int,
):
    """Construct the rollout function.

    The arguments of this function are the unchanging components of the rollout
    function.
    """

    def rollout_fn(env_state, timestep, buffer_state, key):
        """Rollout function.

        The arguments of this function are the changing components of the rollout
        function.
        """

        def step_fn(carry, _):
            env_state, timestep, key = carry

            # Select an action
            key, subkey = jax.random.split(key)
            action_dist = actor_fn(
                dqn_agent_state.q_network_params, timestep.observation
            )
            action = action_dist.sample(seed=subkey)

            # STEP ENVIRONMENT
            next_env_state, next_timestep = jax.vmap(env.step, in_axes=(0, 0))(
                env_state, action
            )

            # LOG EPISODE METRICS
            done = next_timestep.last().reshape(-1)
            info = next_timestep.extras["episode_statistics"]
            next_obs = next_timestep.extras["next_obs"]

            transition = Transition(
                timestep.observation,
                action,
                next_timestep.reward,
                done,
                next_obs,
                info,
            )
            return (next_env_state, next_timestep, key), transition

        # ROLLOUT
        (env_state, timestep, key), transitions = jax.lax.scan(
            step_fn, (env_state, timestep, key), None, length=num_steps
        )

        print("\nTransitions structure:")
        jax.tree_util.tree_map(lambda x: print(f"{x.shape}"), transitions)

        # Attempt to add to buffer
        buffer_state = buffer_add_fn(buffer_state, transitions)
        return env_state, timestep, buffer_state, key

    return rollout_fn


def main():
    train_env, eval_env = create_jumaji_environment(
        "Snake-v1",
        observation_attribute="grid",
        env_kwargs={"num_rows": 6, "num_cols": 6},
    )

    key = jax.random.key(0)

    # CREATE THE BUFFER
    # init_obs = train_env.observation_spec.generate_value()

    # buffer = fbx.make_item_buffer(
    #     max_length=100_000,
    #     min_length=32,
    #     sample_batch_size=32,
    #     add_sequences=True,
    #     add_batches=True,
    # )
    # buffer = buffer.replace(
    #     init=jax.jit(buffer.init),
    #     add=jax.jit(buffer.add, donate_argnums=0),
    #     sample=jax.jit(buffer.sample),
    #     can_sample=jax.jit(buffer.can_sample),
    # )

    # # Initialize buffer functions and state
    # buffer_state = buffer.init(
    #     Transition(
    #         obs=init_obs,
    #         action=jnp.zeros((), dtype=jnp.int32),
    #         next_obs=init_obs,
    #         reward=jnp.zeros((), dtype=jnp.float32),
    #         done=jnp.zeros((), dtype=jnp.bool_),
    #         info={
    #             "episode_return": jnp.zeros((), dtype=jnp.float32),
    #             "episode_length": jnp.zeros((), dtype=jnp.int32),
    #             "is_terminal_step": jnp.zeros((), dtype=jnp.bool_),
    #         },
    #     )
    # )
    # Create replay buffer
    init_x = train_env.observation_spec.generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)
    dummy_transition = Transition(
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        action=jnp.zeros((), dtype=int),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        next_obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        info={"episode_return": 0.0, "episode_length": 0, "is_terminal_step": False},
    )
    buffer = fbx.make_item_buffer(
        max_length=100000,
        min_length=10,
        sample_batch_size=10,
        add_batches=True,
        add_sequences=True,
    )
    buffer_state = buffer.init(dummy_transition)

    # CREATE THE AGENT
    q_network_torso = MLPTorso(layer_sizes=[64, 64])
    q_network_head = DiscreteQNetworkHead(action_dim=4)
    q_network = FeedForwardActor(torso=q_network_torso, action_head=q_network_head)
    params = q_network.init(key, train_env.observation_spec.generate_value())

    # INITIALISE THE ENVIRONMENT
    n_devices = jax.device_count()
    batch_size = 10
    num_envs = 5
    key, *env_keys = jax.random.split(key, n_devices * batch_size * num_envs + 1)
    env_states, timesteps = jax.vmap(train_env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )

    def reshape_states(x):
        return x.reshape((n_devices, batch_size, num_envs) + x.shape[1:])

    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    # PREPARE FOR ROLLOUT
    actor_fn = q_network.apply
    dqn_agent_state = DQNAgentState(
        q_network_params=params,
        opt_state=None,
        buffer_state=buffer_state,
        env_state=env_states,
        timestep=timesteps,
        key=key,
    )

    rollout_fn = create_rollout_fn(
        env=train_env,
        actor_fn=q_network.apply,
        buffer_add_fn=buffer.add,
        dqn_agent_state=dqn_agent_state,
        num_steps=100,
    )

    # 1. Setup the rollout keys to work with pmap + vmap rollout
    key, rollout_key = jax.random.split(key)
    rollout_keys = jnp.stack(jax.random.split(rollout_key, n_devices * batch_size))
    rollout_keys = rollout_keys.reshape(
        (n_devices, batch_size) + rollout_keys.shape[1:]
    )

    # 2. Setup the learner to work with pmap + vmap rollout
    # 2.1 Pack the learner state for duplication (*features, batch)
    learner_state = (dqn_agent_state.q_network_params, dqn_agent_state.buffer_state)

    # 2.2 Duplicate the learner state for each batch (batch_size, *features)
    def broadcast(x):
        # print(f"broadcasting: {x.shape} -> {(batch_size,) + x.shape}")
        return jnp.broadcast_to(x, (batch_size,) + x.shape)

    batched_learner_state = jax.tree_util.tree_map(broadcast, learner_state)

    # 2.3 Replicate the learner state for each device (devices, batch_size, *features)
    replicated_batched_learner_state = flax.jax_utils.replicate(
        batched_learner_state, devices=jax.devices()
    )
    # jax.tree_util.tree_map(lambda x: print(x.shape), replicated_batched_learner_state)

    # 2.4 Update the agent state
    params, buffer_state = replicated_batched_learner_state
    dqn_agent_state = dqn_agent_state._replace(
        q_network_params=params,
        buffer_state=buffer_state,
    )

    # ROLLOUT

    vmap_rollout_fn = jax.vmap(
        rollout_fn,
        in_axes=(0, 0, 0, 0),
        out_axes=(0, 0, 0, 0),
        axis_name="batch",
    )
    pmap_rollout_fn = jax.pmap(vmap_rollout_fn, axis_name="device")

    pmap_rollout_fn(
        dqn_agent_state.env_state,
        dqn_agent_state.timestep,
        dqn_agent_state.buffer_state,
        rollout_keys,
    )


if __name__ == "__main__":
    main()
