from typing import NamedTuple

import chex
import flashbax as fbx
import jax
import jax.numpy as jnp

NUM_ENVS = 5
SEQUENCE_LENGTH = 10


class Observation(NamedTuple):
    agent_view: chex.Array
    action_mask: chex.Array
    step_count: chex.Numeric


# Define a simple transition structure
class Transition(NamedTuple):
    obs: Observation
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: Observation
    info: dict


def main():
    # Create a simple buffer
    buffer = fbx.make_item_buffer(
        max_length=1000,
        min_length=10,
        sample_batch_size=32,
        add_sequences=True,
        add_batches=True,
    )

    init_obs = Observation(
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

    obs_batch = Observation(
        agent_view=jnp.ones((SEQUENCE_LENGTH, NUM_ENVS, 180), dtype=jnp.float32),
        action_mask=jnp.ones((SEQUENCE_LENGTH, NUM_ENVS, 4), dtype=jnp.float32),
        step_count=jnp.ones((SEQUENCE_LENGTH, NUM_ENVS), dtype=jnp.int32),
    )

    # Create a batch of transitions with time dimension
    # Shape: (num_steps, batch_size, ...)
    transitions = Transition(
        obs=obs_batch,
        action=jnp.ones((SEQUENCE_LENGTH, NUM_ENVS), dtype=jnp.int32),
        reward=jnp.ones((SEQUENCE_LENGTH, NUM_ENVS), dtype=jnp.float32),
        next_obs=obs_batch,
        done=jnp.zeros((SEQUENCE_LENGTH, NUM_ENVS), dtype=jnp.bool_),
        info={
            "episode_return": jnp.ones((SEQUENCE_LENGTH, NUM_ENVS), dtype=jnp.float32),
            "episode_length": jnp.ones((SEQUENCE_LENGTH, NUM_ENVS), dtype=jnp.int32),
            "is_terminal_step": jnp.zeros((SEQUENCE_LENGTH, NUM_ENVS), dtype=jnp.bool_),
        },
    )

    jax.tree_util.tree_map(lambda x: print(x.shape), transitions)

    # This is where I'm stuck - how to add these transitions to the buffer?
    buffer_state = buffer.add(buffer_state, transitions)


if __name__ == "__main__":
    main()
