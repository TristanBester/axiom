import jax
from jumanji.env import Environment

from axiom.algorithms.rollout.step import create_step_env_fn
from axiom.algorithms.rollout.types import Transition
from axiom.algorithms.types import DQNAgentState
from axiom.networks.actor import ActorFn


def rollout(
    env: Environment,
    actor_fn: ActorFn,
    dqn_agent_state: DQNAgentState,
    num_steps: int,
) -> tuple[DQNAgentState, Transition]:
    """Rollout the environment for a given number of steps."""
    # Create a step function
    step_env_fn = create_step_env_fn(env, actor_fn, dqn_agent_state)

    # Rollout the policy in the environment
    carry = (dqn_agent_state.env_state, dqn_agent_state.timestep, dqn_agent_state.key)
    (state, timestep, key), transitions = jax.lax.scan(
        step_env_fn, carry, None, length=num_steps
    )

    # Update the DQNAgentState
    dqn_agent_state = dqn_agent_state._replace(
        env_state=state,
        timestep=timestep,
        key=key,
    )
    return dqn_agent_state, transitions
