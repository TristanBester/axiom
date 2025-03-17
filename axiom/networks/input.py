import chex
import flax.linen as nn

from axiom.envs.types import AxiomObservation


class AxiomObservationInput(nn.Module):
    """Extract the agent view from the observation."""

    @nn.compact
    def __call__(self, observation: AxiomObservation) -> chex.Array:
        agent_view = observation.agent_view
        return agent_view
