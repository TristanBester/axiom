import jax.numpy as jnp


def compute_episode_statistics(statistics: dict) -> dict:
    # Extract statistics
    episode_returns = statistics["episode_return"]
    episode_lengths = statistics["episode_length"]
    is_terminal_step = statistics["is_terminal_step"]

    # Mask returns and lengths to only show values at terminal steps
    masked_returns = jnp.where(is_terminal_step, episode_returns, 0.0)
    masked_lengths = jnp.where(is_terminal_step, episode_lengths, 0)

    # Count the number of terminal steps (completed episodes)
    num_terminal_steps = jnp.sum(is_terminal_step)

    # Compute sums
    sum_terminal_returns = jnp.sum(jnp.asarray(masked_returns))
    sum_terminal_lengths = jnp.sum(jnp.asarray(masked_lengths))

    # Calculate averages (avoid division by zero)
    avg_return = jnp.where(
        num_terminal_steps > 0, sum_terminal_returns / num_terminal_steps, 0.0
    )

    avg_length = jnp.where(
        num_terminal_steps > 0, sum_terminal_lengths / num_terminal_steps, 0.0
    )

    return {
        "avg_episode_return": avg_return,
        "avg_episode_length": avg_length,
        "num_episodes": num_terminal_steps,
    }
