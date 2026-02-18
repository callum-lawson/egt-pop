"""Test that maze_dr_v2 produces identical results to maze_dr.

Compares the three core computational functions:
- compute_gae: GAE advantage estimation
- rollout_training_trajectories: rollout collection
- update_actor_critic: PPO parameter update
"""

from functools import partial
import os
import time

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


RUN_SLOW_EQUIVALENCE_TESTS = os.environ.get("RUN_SLOW_EQUIVALENCE_TESTS", "0") == "1"
SMALL_CONFIG = {
    "n_train_envs": 4,
    "n_steps": 8,
    "n_minibatches": 1,
    "n_ppo_epochs": 2,
    "n_updates": 10,
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
    n_train_envs=SMALL_CONFIG["n_train_envs"],
    n_steps=SMALL_CONFIG["n_steps"],
    n_minibatches=SMALL_CONFIG["n_minibatches"],
    n_ppo_epochs=SMALL_CONFIG["n_ppo_epochs"],
    n_updates=SMALL_CONFIG["n_updates"],
    eval_freq=1,
)

OPTIMIZER_CONFIG = v2.OptimizerConfig(
    lr=SMALL_CONFIG["lr"],
    max_grad_norm=SMALL_CONFIG["max_grad_norm"],
)

NETWORK_CONFIG = v2.NetworkConfig()


def _tree_bytes(tree):
    """Return the total byte size for all array leaves in a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    return sum(np.asarray(leaf).nbytes for leaf in leaves)


def _assert_tree_allclose(tree1, tree2, *, rtol=1e-5, atol=1e-8):
    """Assert that two pytrees have numerically close leaves."""
    leaves1 = jax.tree_util.tree_leaves(tree1)
    leaves2 = jax.tree_util.tree_leaves(tree2)
    assert len(leaves1) == len(leaves2)
    for leaf1, leaf2 in zip(leaves1, leaves2):
        assert_allclose(np.asarray(leaf1), np.asarray(leaf2), rtol=rtol, atol=atol)


def _assert_train_state_update_outputs_match(state1, state2, *, rtol=1e-5):
    """Assert update outputs that should be numerically equivalent across implementations."""
    assert state1.step == state2.step
    assert state1.update_count == state2.update_count
    _assert_tree_allclose(state1.params, state2.params, rtol=rtol)
    _assert_tree_allclose(state1.opt_state, state2.opt_state, rtol=rtol)
    _assert_tree_allclose(state1.last_policy_carry, state2.last_policy_carry, rtol=rtol)
    _assert_tree_allclose(state1.last_env_snapshot, state2.last_env_snapshot, rtol=rtol)


def _block_tree(tree):
    """Block on all leaves in a pytree."""
    jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
        tree,
    )


def _run_v1_train_step(rng, env, env_params, train_state):
    """Run one v1 train step body with explicit inputs."""
    n_envs = SMALL_CONFIG["n_train_envs"]
    n_steps = SMALL_CONFIG["n_steps"]
    rng, _ = jax.random.split(rng)
    (
        (rng, train_state, hstate, last_obs, last_env_state, last_value),
        (obs, actions, rewards, dones, log_probs, values, _),
    ) = v1.sample_trajectories_rnn(
        rng,
        env,
        env_params,
        train_state,
        train_state.last_policy_carry,
        train_state.last_env_snapshot.obs,
        train_state.last_env_snapshot.env_state,
        n_envs,
        n_steps,
    )
    advantages, targets = v1.compute_gae(
        SMALL_CONFIG["gamma"],
        SMALL_CONFIG["gae_lambda"],
        last_value,
        values,
        rewards,
        dones,
    )
    (rng, train_state), losses = v1.update_actor_critic_rnn(
        rng,
        train_state,
        train_state.last_policy_carry,
        (obs, actions, dones, log_probs, values, targets, advantages),
        n_envs,
        n_steps,
        SMALL_CONFIG["n_minibatches"],
        SMALL_CONFIG["n_ppo_epochs"],
        SMALL_CONFIG["clip_eps"],
        SMALL_CONFIG["entropy_coeff"],
        SMALL_CONFIG["critic_coeff"],
        update_grad=True,
    )
    train_state = train_state.replace(
        update_count=train_state.update_count + 1,
        last_policy_carry=hstate,
        last_env_snapshot=v2.EnvSnapshot(
            obs=last_obs,
            env_state=last_env_state,
            done=dones[-1],
        ),
    )
    metrics = {"losses": jax.tree_util.tree_map(lambda x: x.mean(), losses)}
    return (rng, train_state), metrics


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
        network_config=NETWORK_CONFIG,
    ))(rng)

    return env_wrapped, env_params, train_state


def test_compute_gae():
    """GAE computation produces identical results with both signatures."""
    rng = jax.random.PRNGKey(0)
    n_steps, n_envs = 16, 8
    keys = jax.random.split(rng, 4)
    values = jax.random.normal(keys[0], (n_steps, n_envs))
    rewards = jax.random.normal(keys[1], (n_steps, n_envs))
    dones = jax.random.bernoulli(keys[2], 0.1, (n_steps, n_envs)).astype(jnp.float32)
    last_value = jax.random.normal(keys[3], (n_envs,))

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


def test_rollout_training_trajectories_and_bootstrap(env_and_state):
    """Rollout sampling and bootstrap value computation match v1 behavior."""
    env, env_params, train_state = env_and_state
    rng = jax.random.PRNGKey(7)
    n_envs = SMALL_CONFIG["n_train_envs"]
    n_steps = SMALL_CONFIG["n_steps"]

    (_, _, hstate1, last_obs1, last_env_state1, last_value_1), traj1 = v1.sample_trajectories_rnn(
        rng, env, env_params, train_state,
        train_state.last_policy_carry,
        train_state.last_env_snapshot.obs,
        train_state.last_env_snapshot.env_state,
        n_envs, n_steps,
    )

    (_, _, policy_carry, env_snapshot), traj2 = v2.rollout_training_trajectories(
        rng, train_state,
        train_state.last_policy_carry, train_state.last_env_snapshot,
        env=env, env_params=env_params,
        train_loop_shape=v2.TrainLoopShape(
            n_train_envs=n_envs,
            n_steps=n_steps,
            n_minibatches=SMALL_CONFIG["n_minibatches"],
            n_ppo_epochs=SMALL_CONFIG["n_ppo_epochs"],
            n_updates=1,
            eval_freq=1,
        ),
    )
    last_value_2 = v2.compute_bootstrap_value(train_state, policy_carry, env_snapshot)

    assert_allclose(np.array(last_value_1), np.array(last_value_2), rtol=1e-5)

    v1_obs, v1_action, v1_reward, v1_done, v1_log_prob, v1_value, _ = traj1
    for l1, l2 in zip(jax.tree_util.tree_leaves(v1_obs), jax.tree_util.tree_leaves(traj2.obs)):
        assert_allclose(np.array(l1), np.array(l2), rtol=1e-5)
    assert_allclose(np.array(v1_action), np.array(traj2.action))
    assert_allclose(np.array(v1_reward), np.array(traj2.reward))
    assert_allclose(np.array(v1_done), np.array(traj2.done))
    assert_allclose(np.array(v1_log_prob), np.array(traj2.log_prob), rtol=1e-5)
    assert_allclose(np.array(v1_value), np.array(traj2.value), rtol=1e-5)
    _assert_tree_allclose(traj1[-1], traj2.info, rtol=1e-5)

    _assert_tree_allclose(hstate1, policy_carry, rtol=1e-5)
    _assert_tree_allclose(last_obs1, env_snapshot.obs, rtol=1e-5)
    _assert_tree_allclose(last_env_state1, env_snapshot.env_state, rtol=1e-5)
    assert_allclose(np.asarray(v1_done[-1]), np.asarray(env_snapshot.done))


def test_update_actor_critic(env_and_state):
    """PPO update produces identical losses and params with both signatures."""
    env, env_params, train_state = env_and_state
    rng = jax.random.PRNGKey(7)
    n_envs = SMALL_CONFIG["n_train_envs"]
    n_steps = SMALL_CONFIG["n_steps"]

    # Generate a shared rollout via v1 (already proven equivalent above)
    (_, _, _, _, _, last_value), (obs, actions, rewards, dones, log_probs, values, info) = \
        v1.sample_trajectories_rnn(
            rng, env, env_params, train_state,
            train_state.last_policy_carry,
            train_state.last_env_snapshot.obs,
            train_state.last_env_snapshot.env_state,
            n_envs, n_steps,
        )
    advantages, targets = v1.compute_gae(
        SMALL_CONFIG["gamma"], SMALL_CONFIG["gae_lambda"],
        last_value, values, rewards, dones,
    )
    batch = (obs, actions, dones, log_probs, values, targets, advantages)

    rng_update = jax.random.PRNGKey(99)

    (rng1, ts1), losses1 = v1.update_actor_critic_rnn(
        rng_update, train_state, train_state.last_policy_carry, batch,
        n_envs, n_steps,
        SMALL_CONFIG["n_minibatches"], SMALL_CONFIG["n_ppo_epochs"],
        SMALL_CONFIG["clip_eps"], SMALL_CONFIG["entropy_coeff"], SMALL_CONFIG["critic_coeff"],
    )

    hparams = v2.PPOHyperparams(
        clip_eps=SMALL_CONFIG["clip_eps"],
        entropy_coeff=SMALL_CONFIG["entropy_coeff"],
        critic_coeff=SMALL_CONFIG["critic_coeff"],
    )
    train_loop_shape = v2.TrainLoopShape(
        n_train_envs=n_envs,
        n_steps=n_steps,
        n_minibatches=SMALL_CONFIG["n_minibatches"],
        n_ppo_epochs=SMALL_CONFIG["n_ppo_epochs"],
        n_updates=1,
        eval_freq=1,
    )
    batch_v2 = v2.PPOUpdateBatch(
        obs=obs,
        action=actions,
        done=dones,
        log_prob=log_probs,
        value=values,
        targets=targets,
        advantages=advantages,
    )
    (rng2, ts2), losses2 = v2.update_actor_critic(
        rng_update, train_state, train_state.last_policy_carry, batch_v2, hparams,
        train_loop_shape=train_loop_shape,
    )

    loss1, (vf1, clip1, ent1) = losses1
    loss2, (vf2, clip2, ent2) = losses2
    assert_allclose(np.array(loss1), np.array(loss2), rtol=1e-5)
    assert_allclose(np.array(vf1), np.array(vf2), rtol=1e-5)
    assert_allclose(np.array(clip1), np.array(clip2), rtol=1e-5)
    assert_allclose(np.array(ent1), np.array(ent2), rtol=1e-5)

    assert_allclose(np.array(rng1), np.array(rng2))
    _assert_train_state_update_outputs_match(ts1, ts2, rtol=1e-5)


def test_prepare_ppo_batch_done_shift_matches_v1():
    """Done-flag shifting in v2 matches v1 PPO batch preparation."""
    rng = jax.random.PRNGKey(123)
    dones = jax.random.bernoulli(rng, 0.25, (11, 5)).astype(jnp.float32)
    batch_v2 = v2.PPOUpdateBatch(
        obs=None,
        action=jnp.zeros((11, 5)),
        done=dones,
        log_prob=jnp.zeros((11, 5)),
        value=jnp.zeros((11, 5)),
        targets=jnp.zeros((11, 5)),
        advantages=jnp.zeros((11, 5)),
    )
    prepared = v2.prepare_ppo_batch(batch_v2)
    shifted_v1 = jnp.roll(dones, 1, axis=0).at[0].set(False)
    assert_allclose(np.asarray(prepared.done), np.asarray(shifted_v1))


def test_build_minibatches_matches_v1_logic(env_and_state):
    """Minibatch shuffling and reshaping in v2 matches the v1 implementation."""
    _, _, train_state = env_and_state
    rng = jax.random.PRNGKey(456)
    n_envs = SMALL_CONFIG["n_train_envs"]
    n_steps = SMALL_CONFIG["n_steps"]
    n_minibatches = SMALL_CONFIG["n_minibatches"]

    keys = jax.random.split(rng, 6)
    permutation = jax.random.permutation(keys[0], n_envs)
    (_, _, _, _, _, _), (obs, _, _, _, _, _, _) = v1.sample_trajectories_rnn(
        keys[1],
        env_and_state[0],
        env_and_state[1],
        train_state,
        train_state.last_policy_carry,
        train_state.last_env_snapshot.obs,
        train_state.last_env_snapshot.env_state,
        n_envs,
        n_steps,
    )
    actions = jax.random.randint(keys[2], (n_steps, n_envs), minval=0, maxval=3)
    dones = jax.random.bernoulli(keys[3], 0.2, (n_steps, n_envs)).astype(jnp.float32)
    log_probs = jax.random.normal(keys[4], (n_steps, n_envs))
    values = jax.random.normal(keys[5], (n_steps, n_envs))
    targets = values + 0.1
    advantages = jax.random.normal(keys[0], (n_steps, n_envs))

    batch_v2 = v2.PPOUpdateBatch(
        obs=obs,
        action=actions,
        done=dones,
        log_prob=log_probs,
        value=values,
        targets=targets,
        advantages=advantages,
    )

    expected = (
        jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0).reshape(n_minibatches, -1, *x.shape[1:]),
            train_state.last_policy_carry,
        ),
        *jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=1)
            .reshape(x.shape[0], n_minibatches, -1, *x.shape[2:])
            .swapaxes(0, 1),
            batch_v2,
        ),
    )
    actual = v2.build_minibatches(train_state.last_policy_carry, batch_v2, permutation, n_minibatches)
    _assert_tree_allclose(expected, actual, rtol=1e-6)


def test_eval_rollout_and_episode_returns_match_v1(env_and_state):
    """Evaluation rollout states, rewards, lengths, and returns match v1 behavior."""
    env, env_params, train_state = env_and_state
    rng = jax.random.PRNGKey(999)
    max_episode_length = 12

    states_v1, rewards_v1, ep_lengths_v1 = v1.evaluate_rnn(
        rng,
        env,
        env_params,
        train_state,
        train_state.last_policy_carry,
        train_state.last_env_snapshot.obs,
        train_state.last_env_snapshot.env_state,
        max_episode_length,
    )
    init_env_snapshot = v2.EnvSnapshot(
        obs=train_state.last_env_snapshot.obs,
        env_state=train_state.last_env_snapshot.env_state,
        done=jnp.zeros(SMALL_CONFIG["n_train_envs"], dtype=bool),
    )
    states_v2, rewards_v2, ep_lengths_v2 = v2.rollout_eval_episodes(
        rng,
        train_state,
        train_state.last_policy_carry,
        init_env_snapshot,
        env=env,
        env_params=env_params,
        max_episode_length=max_episode_length,
    )
    returns_v2 = v2.compute_episode_returns(
        rewards_v2,
        ep_lengths_v2,
        max_episode_length=max_episode_length,
    )
    returns_v1 = (
        rewards_v1
        * (jnp.arange(max_episode_length)[..., None] < ep_lengths_v1)
    ).sum(axis=0)

    _assert_tree_allclose(states_v1, states_v2, rtol=1e-5)
    assert_allclose(np.asarray(rewards_v1), np.asarray(rewards_v2), rtol=1e-5)
    assert_allclose(np.asarray(ep_lengths_v1), np.asarray(ep_lengths_v2))
    assert_allclose(np.asarray(returns_v1), np.asarray(returns_v2), rtol=1e-5)


def test_train_step_equivalence_with_aligned_rng(env_and_state):
    """One training step in v2 matches v1 when both use the same effective RNG stream."""
    env, env_params, train_state = env_and_state
    input_rng = jax.random.PRNGKey(31415)

    (rng_v1, state_v1), metrics_v1 = _run_v1_train_step(input_rng, env, env_params, train_state)

    hparams = v2.PPOHyperparams(
        gamma=SMALL_CONFIG["gamma"],
        gae_lambda=SMALL_CONFIG["gae_lambda"],
        clip_eps=SMALL_CONFIG["clip_eps"],
        entropy_coeff=SMALL_CONFIG["entropy_coeff"],
        critic_coeff=SMALL_CONFIG["critic_coeff"],
    )
    aligned_rng, _ = jax.random.split(input_rng)
    (rng_v2, state_v2), metrics_v2 = v2.train_step(
        (aligned_rng, train_state),
        None,
        hparams=hparams,
        env=env,
        env_params=env_params,
        train_loop_shape=TRAIN_LOOP_SHAPE,
    )

    assert_allclose(np.asarray(rng_v1), np.asarray(rng_v2))
    assert state_v1.update_count == state_v2.update_count
    _assert_tree_allclose(state_v1.params, state_v2.params, rtol=1e-5)
    _assert_tree_allclose(state_v1.opt_state, state_v2.opt_state, rtol=1e-5)
    _assert_tree_allclose(state_v1.last_policy_carry, state_v2.last_policy_carry, rtol=1e-5)
    _assert_tree_allclose(state_v1.last_env_snapshot, state_v2.last_env_snapshot, rtol=1e-5)
    _assert_tree_allclose(metrics_v1["losses"], metrics_v2["losses"], rtol=1e-5)


@pytest.mark.skipif(
    not RUN_SLOW_EQUIVALENCE_TESTS,
    reason="Set RUN_SLOW_EQUIVALENCE_TESTS=1 to run timing equivalence checks.",
)
def test_v2_rollout_update_runtime_is_within_tolerance(env_and_state):
    """V2 rollout+update runtime should stay within a modest ratio of v1 runtime."""
    env, env_params, train_state = env_and_state
    n_envs = SMALL_CONFIG["n_train_envs"]
    n_steps = SMALL_CONFIG["n_steps"]
    n_iters = 8

    hparams = v2.PPOHyperparams(
        gamma=SMALL_CONFIG["gamma"],
        gae_lambda=SMALL_CONFIG["gae_lambda"],
        clip_eps=SMALL_CONFIG["clip_eps"],
        entropy_coeff=SMALL_CONFIG["entropy_coeff"],
        critic_coeff=SMALL_CONFIG["critic_coeff"],
    )

    @jax.jit
    def v1_rollout_update_step(rng, current_state):
        (_, _, _, _, _, last_value), (obs, actions, rewards, dones, log_probs, values, _) = v1.sample_trajectories_rnn(
            rng,
            env,
            env_params,
            current_state,
            current_state.last_policy_carry,
            current_state.last_env_snapshot.obs,
            current_state.last_env_snapshot.env_state,
            n_envs,
            n_steps,
        )
        advantages, targets = v1.compute_gae(
            SMALL_CONFIG["gamma"],
            SMALL_CONFIG["gae_lambda"],
            last_value,
            values,
            rewards,
            dones,
        )
        (next_rng, next_state), losses = v1.update_actor_critic_rnn(
            rng,
            current_state,
            current_state.last_policy_carry,
            (obs, actions, dones, log_probs, values, targets, advantages),
            n_envs,
            n_steps,
            SMALL_CONFIG["n_minibatches"],
            SMALL_CONFIG["n_ppo_epochs"],
            SMALL_CONFIG["clip_eps"],
            SMALL_CONFIG["entropy_coeff"],
            SMALL_CONFIG["critic_coeff"],
        )
        return next_rng, next_state, jnp.mean(losses[0])

    @jax.jit
    def v2_rollout_update_step(rng, current_state):
        (_, _, policy_carry, env_snapshot), trajectory = v2.rollout_training_trajectories(
            rng,
            current_state,
            current_state.last_policy_carry,
            current_state.last_env_snapshot,
            env=env,
            env_params=env_params,
            train_loop_shape=TRAIN_LOOP_SHAPE,
        )
        last_value = v2.compute_bootstrap_value(current_state, policy_carry, env_snapshot)
        advantages, targets = v2.compute_gae(hparams, last_value, trajectory)
        batch = v2.build_ppo_update_batch(trajectory, targets, advantages)
        (next_rng, next_state), losses = v2.update_actor_critic(
            rng,
            current_state,
            current_state.last_policy_carry,
            batch,
            hparams,
            train_loop_shape=TRAIN_LOOP_SHAPE,
        )
        return next_rng, next_state, jnp.mean(losses[0])

    def _measure_seconds(step_fn, seed):
        rng = jax.random.PRNGKey(seed)
        state = train_state
        rng, state, scalar = step_fn(rng, state)
        scalar.block_until_ready()
        start = time.perf_counter()
        for _ in range(n_iters):
            rng, state, scalar = step_fn(rng, state)
        scalar.block_until_ready()
        return time.perf_counter() - start

    v1_seconds = _measure_seconds(v1_rollout_update_step, seed=11)
    v2_seconds = _measure_seconds(v2_rollout_update_step, seed=11)

    ratio = v2_seconds / v1_seconds
    assert ratio < 1.5, f"v2 runtime ratio too high: {ratio:.3f} (v1={v1_seconds:.3f}s, v2={v2_seconds:.3f}s)"


@pytest.mark.skipif(
    not RUN_SLOW_EQUIVALENCE_TESTS,
    reason="Set RUN_SLOW_EQUIVALENCE_TESTS=1 to run compute-footprint checks.",
)
def test_v2_rollout_update_tensor_footprint_matches_v1(env_and_state):
    """V2 rollout/update tensor footprint should match v1 for identical shapes."""
    env, env_params, train_state = env_and_state
    rng = jax.random.PRNGKey(222)
    n_envs = SMALL_CONFIG["n_train_envs"]
    n_steps = SMALL_CONFIG["n_steps"]

    (_, _, _, _, _, _), traj1 = v1.sample_trajectories_rnn(
        rng,
        env,
        env_params,
        train_state,
        train_state.last_policy_carry,
        train_state.last_env_snapshot.obs,
        train_state.last_env_snapshot.env_state,
        n_envs,
        n_steps,
    )
    (_, _, policy_carry, env_snapshot), traj2 = v2.rollout_training_trajectories(
        rng,
        train_state,
        train_state.last_policy_carry,
        train_state.last_env_snapshot,
        env=env,
        env_params=env_params,
        train_loop_shape=TRAIN_LOOP_SHAPE,
    )
    _block_tree((traj1, traj2, policy_carry, env_snapshot))

    traj1_bytes = _tree_bytes(traj1)
    traj2_bytes = _tree_bytes(traj2)
    params1_bytes = _tree_bytes(train_state.params)
    params2_bytes = _tree_bytes(train_state.params)

    assert traj1_bytes == traj2_bytes
    assert params1_bytes == params2_bytes
