import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"


import flashbax as fbx
import jax
import jax.numpy as jnp
from jumanji.env import Environment

from axiom.algorithms.rollout.rollout import rollout
from axiom.algorithms.rollout.types import Transition
from axiom.algorithms.types import DQNAgentState
from axiom.envs.factories import create_jumaji_environment
from axiom.envs.types import AxiomObservation
from axiom.networks.actor import FeedForwardActor
from axiom.networks.head import DiscreteQNetworkHead
from axiom.networks.torso import MLPTorso

NUM_ENVS = 5
SEQUENCE_LENGTH = 10


# def create_buffer(env: Environment):
#     """Create a replay buffer for storing transitions."""
#     init_obs = env.observation_spec.generate_value()

#     buffer = fbx.make_item_buffer(
#         max_length=100_000,
#         min_length=32,
#         sample_batch_size=32,
#         add_sequences=True,
#         add_batches=True,
#     )
#     buffer = buffer.replace(
#         init=jax.jit(buffer.init),
#         add=jax.jit(buffer.add, donate_argnums=0),
#         sample=jax.jit(buffer.sample),
#         can_sample=jax.jit(buffer.can_sample),
#     )

#     # Initialize buffer functions and state
#     buffer_state = buffer.init(
#         Transition(
#             obs=init_obs,
#             action=jnp.zeros((), dtype=jnp.int32),
#             next_obs=init_obs,
#             reward=jnp.zeros((), dtype=jnp.float32),
#             done=jnp.zeros((), dtype=jnp.bool_),
#             info={
#                 "episode_return": jnp.zeros((), dtype=jnp.float32),
#                 "episode_length": jnp.zeros((), dtype=jnp.int32),
#                 "is_terminal_step": jnp.zeros((), dtype=jnp.bool_),
#             },
#         )
#     )

#     init_x = env.observation_spec.generate_value()
#     init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

#     # Create replay buffer
#     dummy_transition = Transition(
#         obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
#         action=jnp.zeros((), dtype=int),
#         reward=jnp.zeros((), dtype=float),
#         done=jnp.zeros((), dtype=bool),
#         next_obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
#         info={"episode_return": 0.0, "episode_length": 0, "is_terminal_step": False},
#     )
#     buffer = fbx.make_item_buffer(
#         max_length=100_000,
#         min_length=32,
#         sample_batch_size=32,
#         add_batches=True,
#         add_sequences=True,
#     )
#     buffer_state = buffer.init(dummy_transition)
#     return buffer, buffer_state


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
    #             "episode_return": 0.0,
    #             "episode_length": 0,
    #             "is_terminal_step": False,
    #         },
    #     )
    # )

    buffer = fbx.make_item_buffer(
        max_length=1000,
        min_length=10,
        sample_batch_size=32,
        add_sequences=True,
        add_batches=True,
    )

    init_obs = AxiomObservation(
        agent_view=jnp.zeros((180,), dtype=jnp.float32),
        action_mask=jnp.zeros((4,), dtype=jnp.float32),
        step_count=jnp.zeros((), dtype=jnp.int32),
    )

    # Initialize the buffer with a dummy transition
    buffer_state = buffer.init(
        Transition(
            obs=init_obs,
            action=jnp.zeros((), dtype=jnp.int32),
            reward=jnp.zeros((), dtype=jnp.float32),
            next_obs=init_obs,
            done=jnp.zeros((), dtype=jnp.bool_),
            info={
                "episode_return": 0.0,
                "episode_length": 0,
                "is_terminal_step": False,
            },
        )
    )

    # NETWORK
    q_network_torso = MLPTorso(layer_sizes=[64, 64])
    q_network_head = DiscreteQNetworkHead(action_dim=4)
    q_network = FeedForwardActor(torso=q_network_torso, action_head=q_network_head)
    params = q_network.init(key, train_env.observation_spec.generate_value())

    actor_fn = q_network.apply
    dqn_agent_state = DQNAgentState(
        q_network_params=params,
        opt_state=None,
        buffer_state=buffer_state,
        env_state=env_states,
        timestep=timesteps,
        key=subkey,
    )

    # ROLLOUT
    dqn_agent_state = rollout(
        env=train_env,
        actor_fn=actor_fn,
        buffer_add_fn=buffer.add,
        dqn_agent_state=dqn_agent_state,
        num_steps=SEQUENCE_LENGTH,
    )


if __name__ == "__main__":
    main()
