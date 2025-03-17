import jumanji
from jumanji.env import Environment
from jumanji.wrappers import AutoResetWrapper

from axiom.envs.wrappers.conversion import JumanjiToAxiomWrapper
from axiom.envs.wrappers.logging import RecordEpisodeMetricsWrapper
from axiom.envs.wrappers.observation.flatten import FlattenAgentViewWrapper


def create_jumaji_environment(
    name: str, observation_attribute: str | None = None, env_kwargs: dict | None = None
) -> tuple[Environment, Environment]:
    """Create Jumanji environments for training and evaluation.

    Both environments are wrapped with the `JumanjiToAxiomWrapper`. Additionally,
    the training environment is wrapped with the `AutoResetWrapper` and
    `RecordEpisodeMetricsWrapper` to record episode metrics.

    Args:
        name: The name of the environment to create.
        observation_attribute: The attribute of the observation to flatten.
        env_kwargs: The keyword arguments to pass to the environment constructor.

    Returns:
        A tuple of the training and evaluation environments.
    """
    env_kwargs = env_kwargs or {}
    train_env = jumanji.make(name, **env_kwargs)
    eval_env = jumanji.make(name, **env_kwargs)

    # Wrap environments with AxiomObservationWrapper
    train_env = JumanjiToAxiomWrapper(train_env, observation_attribute)
    eval_env = JumanjiToAxiomWrapper(eval_env, observation_attribute)

    # TODO: These optional wrappers should be defined in the config and optionally
    # applied at the end of the factory function.
    train_env = FlattenAgentViewWrapper(train_env)
    eval_env = FlattenAgentViewWrapper(eval_env)

    # Wrap training environment for auto-resetting & logging
    train_env = AutoResetWrapper(train_env, next_obs_in_extras=True)
    train_env = RecordEpisodeMetricsWrapper(train_env)
    return train_env, eval_env
