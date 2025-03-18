import chex
from jumanji.env import Environment

from axiom.networks.actor import FeedForwardActor
from axiom.networks.head import DiscreteQNetworkHead
from axiom.networks.torso import MLPTorso


def network_factory(env: Environment, key: chex.PRNGKey):
    """Create a Q-network for the environment."""
    q_network_torso = MLPTorso(layer_sizes=[64, 64])
    q_network_head = DiscreteQNetworkHead(action_dim=4)
    q_network = FeedForwardActor(torso=q_network_torso, action_head=q_network_head)

    params = q_network.init(key, env.observation_spec.generate_value())
    target_params = params
    return q_network, params, target_params
