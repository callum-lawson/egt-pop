import jax
from flax import traverse_util

from examples import maze_dr_v2 as v2
from jaxued.environments import Maze
from jaxued.environments.maze import make_level_generator
from jaxued.wrappers import AutoResetWrapper


EXPECTED_ACTOR_CRITIC_PARAM_KEY_PATHS = {
    "Conv_0/bias",
    "Conv_0/kernel",
    "OptimizedLSTMCell_0/hf/bias",
    "OptimizedLSTMCell_0/hf/kernel",
    "OptimizedLSTMCell_0/hg/bias",
    "OptimizedLSTMCell_0/hg/kernel",
    "OptimizedLSTMCell_0/hi/bias",
    "OptimizedLSTMCell_0/hi/kernel",
    "OptimizedLSTMCell_0/ho/bias",
    "OptimizedLSTMCell_0/ho/kernel",
    "OptimizedLSTMCell_0/if/kernel",
    "OptimizedLSTMCell_0/ig/kernel",
    "OptimizedLSTMCell_0/ii/kernel",
    "OptimizedLSTMCell_0/io/kernel",
    "actor0/bias",
    "actor0/kernel",
    "actor1/bias",
    "actor1/kernel",
    "critic0/bias",
    "critic0/kernel",
    "critic1/bias",
    "critic1/kernel",
    "scalar_embed/bias",
    "scalar_embed/kernel",
}


def test_actor_critic_param_key_paths_are_checkpoint_compatible():
    """ActorCritic parameter key paths stay stable for checkpoint loading."""
    env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    sample_random_level = make_level_generator(env.max_height, env.max_width, 25)
    env_wrapped = AutoResetWrapper(env, sample_random_level)
    env_params = env_wrapped.default_params
    network_config = v2.NetworkConfig()
    n_train_envs = 4
    initial_policy_carry = v2.ActorCritic.initialize_carry((n_train_envs,), network_config)

    _, network_params = v2.initialize_actor_critic_network(
        jax.random.PRNGKey(0),
        env=env_wrapped,
        env_params=env_params,
        sample_random_level=sample_random_level,
        n_train_envs=n_train_envs,
        network_config=network_config,
        initial_policy_carry=initial_policy_carry,
    )

    flattened_params = traverse_util.flatten_dict(network_params["params"])
    actual_param_key_paths = {"/".join(path_parts) for path_parts in flattened_params}

    assert actual_param_key_paths == EXPECTED_ACTOR_CRITIC_PARAM_KEY_PATHS
