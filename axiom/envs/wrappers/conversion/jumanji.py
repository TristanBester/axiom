from functools import cached_property

import chex
import jax.numpy as jnp
from jumanji.env import Environment, State
from jumanji.specs import Array, Spec
from jumanji.types import Observation, TimeStep
from jumanji.wrappers import Wrapper

from axiom.envs.types import AxiomObservation


class JumanjiToAxiomWrapper(Wrapper):
    """Convert a Jumanji environment to an Axiom environment.

    Main purpose is to convert the TimeStep.observation to an AxiomObservation.

    Args:
        env: The Jumanji environment to wrap.
        observation_attribute: The attribute of the observation to use for the agent
            view. If None, the entire observation is used.
    """

    def __init__(self, env: Environment, observation_attribute: str | None = None):
        super().__init__(env)
        self.observation_attribute = observation_attribute

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[AxiomObservation]]:
        state, timestep = self._env.reset(key)

        action_mask = self._get_action_mask(timestep)
        agent_view = self._get_agent_view(timestep)
        obs = AxiomObservation(
            agent_view=agent_view,
            action_mask=action_mask,
            step_count=state.step_count,
        )

        extras = timestep.extras if timestep.extras is not None else {}
        timestep = timestep.replace(observation=obs, extras=extras)  # type: ignore
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> tuple[State, TimeStep[AxiomObservation]]:
        state, timestep = self._env.step(state, action)

        action_mask = self._get_action_mask(timestep)
        agent_view = self._get_agent_view(timestep)
        obs = AxiomObservation(
            agent_view=agent_view,
            action_mask=action_mask,
            step_count=state.step_count,  # type: ignore
        )

        extras = timestep.extras if timestep.extras is not None else {}
        timestep = timestep.replace(observation=obs, extras=extras)  # type: ignore
        return state, timestep

    @cached_property
    def observation_spec(self) -> Spec:
        """Override environment observation spec to return AxiomObservation objects."""
        # Compute agent_view spec
        if self.observation_attribute is not None:
            agent_view_shape = self._env.observation_spec[
                self.observation_attribute
            ].shape
            agent_view_spec = Array(
                shape=agent_view_shape,
                dtype=jnp.float32,
                name="agent_view",
            )
        else:
            agent_view_spec = self._env.observation_spec

        # Compute action_mask spec
        action_mask_spec = Array(
            shape=(self._env.action_spec.num_values,),
            dtype=jnp.float32,
            name="action_mask",
        )

        # Compute step_count spec
        step_count_spec = Array(shape=(), dtype=jnp.int32, name="step_count")
        return Spec(
            constructor=AxiomObservation,
            name="AxiomObservationSpec",
            agent_view=agent_view_spec,
            action_mask=action_mask_spec,
            step_count=step_count_spec,
        )

    def _get_action_mask(self, timestep: TimeStep[Observation]) -> chex.Array:
        if hasattr(timestep.observation, "action_mask"):
            return timestep.observation.action_mask.astype(jnp.float32)  # type: ignore
        else:
            return jnp.ones(self._env.action_spec.num_values, dtype=jnp.float32)

    def _get_agent_view(self, timestep: TimeStep[Observation]) -> chex.Array:
        if self.observation_attribute is not None:
            return timestep.observation._asdict()[self.observation_attribute].astype(  # type: ignore
                jnp.float32
            )  # type: ignore
        else:
            return timestep.observation.agent_view.astype(jnp.float32)  # type: ignore
