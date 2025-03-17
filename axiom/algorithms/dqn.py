import jax

from axiom.algorithms.rollout.rollout import rollout
from axiom.algorithms.types import DQNAgentState
from axiom.envs.factories import create_jumaji_environment
from axiom.networks.actor import FeedForwardActor
from axiom.networks.head import DiscreteQNetworkHead
from axiom.networks.torso import MLPTorso


def main():
    train_env, eval_env = create_jumaji_environment(
        "Snake-v1",
        observation_attribute="grid",
        env_kwargs={"num_rows": 6, "num_cols": 6},
    )

    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    subkeys = jax.random.split(subkey, 4)
    state, timestep = jax.vmap(train_env.reset)(key=subkeys)

    q_network_torso = MLPTorso(layer_sizes=[64, 64])
    q_network_head = DiscreteQNetworkHead(action_dim=4)
    q_network = FeedForwardActor(torso=q_network_torso, action_head=q_network_head)
    params = q_network.init(key, train_env.observation_spec.generate_value())

    actor_fn = q_network.apply
    dqn_agent_state = DQNAgentState(
        q_network_params=params,
        opt_state=None,
        buffer_state=None,
        env_state=state,
        timestep=timestep,
        key=subkey,
    )

    dqn_agent_state, transitions = rollout(
        env=train_env,
        actor_fn=actor_fn,
        dqn_agent_state=dqn_agent_state,
        num_steps=100,
    )

    breakpoint()


if __name__ == "__main__":
    main()
