"""Test that maze_dr_v2 produces identical results to maze_dr.

Compares the three core computational functions:
- compute_gae: GAE advantage estimation
- sample_trajectories_rnn: rollout collection
- update_actor_critic_rnn: PPO parameter update
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from examples import maze_dr as v1
from examples import maze_dr_v2 as v2
from jaxued.environments import Maze
from jaxued.environments.maze import make_level_generator
from jaxued.wrappers import AutoResetWrapper


SMALL_CONFIG = {
    "num_train_envs": 4,
    "num_steps": 8,
    "num_minibatches": 1,
    "epoch_ppo": 2,
    "num_updates": 10,
    "lr": 1e-4,
    "max_grad_norm": 0.5,
    "gamma": 0.995,
    "gae_lambda": 0.98,
    "clip_eps": 0.2,
    "entropy_coeff": 1e-3,
    "critic_coeff": 0.5,
    "agent_view_size": 5,
    "n_walls": 25,
}

TRAIN_LOOP_SHAPE = v2.TrainLoopShape(
    num_train_envs=SMALL_CONFIG["num_train_envs"],
    num_steps=SMALL_CONFIG["num_steps"],
    num_minibatches=SMALL_CONFIG["num_minibatches"],
    epoch_ppo=SMALL_CONFIG["epoch_ppo"],
    num_updates=SMALL_CONFIG["num_updates"],
    eval_freq=1,
)

OPTIMIZER_CONFIG = v2.OptimizerConfig(
    lr=SMALL_CONFIG["lr"],
    max_grad_norm=SMALL_CONFIG["max_grad_norm"],
)


@pytest.fixture(scope="module")
def env_and_state():
    """Create env, params, and train state shared by all tests."""
    env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    sample_random_level = make_level_generator(env.max_height, env.max_width, 25)
    env_wrapped = AutoResetWrapper(env, sample_random_level)
    env_params = env_wrapped.default_params

    rng = jax.random.PRNGKey(42)
    train_state = jax.jit(partial(
        v2.create_train_state,
        env=env_wrapped,
        env_params=env_params,
        sample_random_level=sample_random_level,
        train_loop_shape=TRAIN_LOOP_SHAPE,
        optimizer_config=OPTIMIZER_CONFIG,
    ))(rng)

    return env_wrapped, env_params, train_state


def test_compute_gae():
    """GAE computation produces identical results with both signatures."""
    rng = jax.random.PRNGKey(0)
    num_steps, num_envs = 16, 8
    keys = jax.random.split(rng, 4)
    values = jax.random.normal(keys[0], (num_steps, num_envs))
    rewards = jax.random.normal(keys[1], (num_steps, num_envs))
    dones = jax.random.bernoulli(keys[2], 0.1, (num_steps, num_envs)).astype(jnp.float32)
    last_value = jax.random.normal(keys[3], (num_envs,))

    gamma, gae_lambda = 0.995, 0.98

    adv1, tgt1 = v1.compute_gae(gamma, gae_lambda, last_value, values, rewards, dones)

    hparams = v2.PPOHyperparams(gamma=gamma, gae_lambda=gae_lambda)
    traj = v2.Trajectory(
        obs=None, action=None, reward=rewards,
        done=dones, log_prob=None, value=values, info=None,
    )
    adv2, tgt2 = v2.compute_gae(hparams, last_value, traj)

    assert_allclose(np.array(adv1), np.array(adv2), rtol=1e-5)
    assert_allclose(np.array(tgt1), np.array(tgt2), rtol=1e-5)


def test_sample_trajectories_rnn(env_and_state):
    """Trajectory sampling produces identical results with both signatures."""
    env, env_params, train_state = env_and_state
    rng = jax.random.PRNGKey(7)
    num_envs = SMALL_CONFIG["num_train_envs"]
    num_steps = SMALL_CONFIG["num_steps"]

    (_, _, _, _, _, last_value_1), traj1 = v1.sample_trajectories_rnn(
        rng, env, env_params, train_state,
        train_state.last_hstate, train_state.last_obs, train_state.last_env_state,
        num_envs, num_steps,
    )

    (_, _, _, _, _, last_value_2), traj2 = v2.sample_trajectories_rnn(
        rng, train_state,
        train_state.last_hstate, train_state.last_obs, train_state.last_env_state,
        env=env, env_params=env_params,
        num_envs=num_envs, max_episode_length=num_steps,
    )

    assert_allclose(np.array(last_value_1), np.array(last_value_2), rtol=1e-5)

    v1_obs, v1_action, v1_reward, v1_done, v1_log_prob, v1_value, _ = traj1
    for l1, l2 in zip(jax.tree_util.tree_leaves(v1_obs), jax.tree_util.tree_leaves(traj2.obs)):
        assert_allclose(np.array(l1), np.array(l2), rtol=1e-5)
    assert_allclose(np.array(v1_action), np.array(traj2.action))
    assert_allclose(np.array(v1_reward), np.array(traj2.reward))
    assert_allclose(np.array(v1_done), np.array(traj2.done))
    assert_allclose(np.array(v1_log_prob), np.array(traj2.log_prob), rtol=1e-5)
    assert_allclose(np.array(v1_value), np.array(traj2.value), rtol=1e-5)


def test_update_actor_critic_rnn(env_and_state):
    """PPO update produces identical losses and params with both signatures."""
    env, env_params, train_state = env_and_state
    rng = jax.random.PRNGKey(7)
    num_envs = SMALL_CONFIG["num_train_envs"]
    num_steps = SMALL_CONFIG["num_steps"]

    # Generate a shared rollout via v1 (already proven equivalent above)
    (_, _, _, _, _, last_value), (obs, actions, rewards, dones, log_probs, values, info) = \
        v1.sample_trajectories_rnn(
            rng, env, env_params, train_state,
            train_state.last_hstate, train_state.last_obs, train_state.last_env_state,
            num_envs, num_steps,
        )
    advantages, targets = v1.compute_gae(
        SMALL_CONFIG["gamma"], SMALL_CONFIG["gae_lambda"],
        last_value, values, rewards, dones,
    )
    batch = (obs, actions, dones, log_probs, values, targets, advantages)

    rng_update = jax.random.PRNGKey(99)

    (rng1, ts1), losses1 = v1.update_actor_critic_rnn(
        rng_update, train_state, train_state.last_hstate, batch,
        num_envs, num_steps,
        SMALL_CONFIG["num_minibatches"], SMALL_CONFIG["epoch_ppo"],
        SMALL_CONFIG["clip_eps"], SMALL_CONFIG["entropy_coeff"], SMALL_CONFIG["critic_coeff"],
    )

    hparams = v2.PPOHyperparams(
        clip_eps=SMALL_CONFIG["clip_eps"],
        entropy_coeff=SMALL_CONFIG["entropy_coeff"],
        critic_coeff=SMALL_CONFIG["critic_coeff"],
    )
    batch_v2 = v2.PPOUpdateBatch(
        obs=obs,
        actions=actions,
        dones=dones,
        log_probs=log_probs,
        values=values,
        targets=targets,
        advantages=advantages,
    )
    (rng2, ts2), losses2 = v2.update_actor_critic_rnn(
        rng_update, train_state, train_state.last_hstate, batch_v2, hparams,
        num_envs=num_envs, n_steps=num_steps,
        n_minibatch=SMALL_CONFIG["num_minibatches"],
        n_epochs=SMALL_CONFIG["epoch_ppo"],
    )

    loss1, (vf1, clip1, ent1) = losses1
    loss2, (vf2, clip2, ent2) = losses2
    assert_allclose(np.array(loss1), np.array(loss2), rtol=1e-5)
    assert_allclose(np.array(vf1), np.array(vf2), rtol=1e-5)
    assert_allclose(np.array(clip1), np.array(clip2), rtol=1e-5)
    assert_allclose(np.array(ent1), np.array(ent2), rtol=1e-5)

    assert_allclose(np.array(rng1), np.array(rng2))
    for p1, p2 in zip(jax.tree_util.tree_leaves(ts1.params), jax.tree_util.tree_leaves(ts2.params)):
        assert_allclose(np.array(p1), np.array(p2), rtol=1e-5)
    for l1, l2 in zip(jax.tree_util.tree_leaves(ts1.opt_state), jax.tree_util.tree_leaves(ts2.opt_state)):
        assert_allclose(np.array(l1), np.array(l2), rtol=1e-5)
