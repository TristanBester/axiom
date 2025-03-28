import chex
import optax


def optim_factory(params: chex.ArrayTree):
    """Create an optimizer for the Q-network."""
    optim = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(4e-4, eps=1e-5),
    )
    optim_state = optim.init(params)
    return optim, optim_state
