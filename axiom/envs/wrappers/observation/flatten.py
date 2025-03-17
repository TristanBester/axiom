from functools import cached_property
from typing import Any

import chex
import jax.numpy as jnp
from jumanji.env import Environment, State
from jumanji.specs import Array, Spec
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from axiom.envs.types import AxiomObservation


class FlattenAgentViewWrapper(Wrapper):
    """Wrapper that flattens the agent view of the observation.

    Wrapper requires the environment to be an Axiom environment.
    """

    def __init__(self, env: Environment[Any, Any, AxiomObservation]):
        super().__init__(env)
        self._original_obs_shape = self._env.observation_spec.agent_view.shape
        self._flattened_obs_shape = (jnp.prod(jnp.asarray(self._original_obs_shape)),)

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[AxiomObservation]]:
        state, timestep = self._env.reset(key)
        timestep = self._flatten(timestep)

        if "next_obs" in timestep.extras:
            timestep.extras["next_obs"] = self._flatten(timestep.extras["next_obs"])
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> tuple[State, TimeStep[AxiomObservation]]:
        state, timestep = self._env.step(state, action)
        timestep = self._flatten(timestep)

        if "next_obs" in timestep.extras:
            timestep.extras["next_obs"] = self._flatten(timestep.extras["next_obs"])
        return state, timestep

    @cached_property
    def observation_spec(self) -> Spec:
        flattened_agent_view_spec = Array(
            shape=self._flattened_obs_shape, dtype=jnp.float32
        )
        modified_observation_spec = self._env.observation_spec.replace(
            agent_view=flattened_agent_view_spec
        )
        return modified_observation_spec

    def _flatten(
        self, timestep: TimeStep[AxiomObservation]
    ) -> TimeStep[AxiomObservation]:
        # Flatten the agent_view in the TimeStep
        original_agent_view = timestep.observation.agent_view
        flattened_agent_view = self._flatten_agent_view(original_agent_view)

        # Replace the agent_view in the TimeStep with the flattened agent_view
        observation = timestep.observation._replace(agent_view=flattened_agent_view)
        timestep = timestep.replace(observation=observation)
        return timestep

    def _flatten_agent_view(self, agent_view: chex.Array) -> chex.Array:
        flattened_agent_view = jnp.reshape(
            agent_view,
            self._flattened_obs_shape,
        )
        return flattened_agent_view.astype(jnp.float32)
