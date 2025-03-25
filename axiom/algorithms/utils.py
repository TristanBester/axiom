import jax
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


# def compute_episode_statistics(metrics):
#     """Get the metrics for the final step of an episode and check if th.

#     Note: this is not a jittable method. We need to return variable length arrays, since
#     we don't know how many episodes have been run. This is done since the logger
#     expects arrays for computing summary statistics on the episode metrics.
#     """
#     is_final_ep = metrics.pop("is_terminal_step")
#     has_final_ep_step = bool(jnp.any(is_final_ep))

#     # If it didn't make it to the final step, return zeros.
#     if not has_final_ep_step:
#         final_metrics = jax.tree_util.tree_map(jnp.zeros_like, metrics)
#     else:
#         final_metrics = jax.tree_util.tree_map(lambda x: x[is_final_ep], metrics)

#     # breakpoint()

#     return final_metrics, has_final_ep_step
