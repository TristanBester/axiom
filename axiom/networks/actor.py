from typing import Any, Protocol

import distrax
import flax.linen as nn

from axiom.envs.types import AxiomObservation
from axiom.networks.input import AxiomObservationInput


class ActorFn(Protocol):
    def __call__(
        self, variables: Any, observation: AxiomObservation
    ) -> distrax.DistributionLike:
        """Return a state conditioned action distribution."""


class FeedForwardActor(nn.Module):
    """Simple feed forward actor network."""

    torso: nn.Module
    action_head: nn.Module
    input_layer: nn.Module = AxiomObservationInput()

    @nn.compact
    def __call__(self, observation: AxiomObservation) -> distrax.DistributionLike:
        agent_view = self.input_layer(observation)
        embedding = self.torso(agent_view)
        return self.action_head(embedding)
