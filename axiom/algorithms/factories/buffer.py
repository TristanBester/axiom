import flashbax as fbx
import jax
import jax.numpy as jnp
from jumanji.env import Environment

from axiom.algorithms.rollout.types import Transition


def buffer_factory(env: Environment):
    """Create a replay buffer for storing transitions."""
    init_obs = env.observation_spec.generate_value()

    buffer = fbx.make_item_buffer(
        max_length=100_000,
        min_length=32,
        sample_batch_size=32,
        add_sequences=True,
        add_batches=True,
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )

    # Initialize buffer functions and state
    buffer_state = buffer.init(
        Transition(
            obs=init_obs,
            action=jnp.zeros((), dtype=jnp.int32),
            next_obs=init_obs,
            reward=jnp.zeros((), dtype=jnp.float32),
            done=jnp.zeros((), dtype=jnp.bool_),
            info={
                "episode_return": 0.0,
                "episode_length": 0,
                "is_terminal_step": False,
            },
        )
    )
    return buffer, buffer_state
