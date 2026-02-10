import os
import json
import time
from functools import partial
from typing import NamedTuple, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import orbax.checkpoint as ocp
import wandb
from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.linen import ResetRNN
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator
from jaxued.utils import max_mc, positive_value_loss
from jaxued.wrappers import AutoResetWrapper
import chex
from config_utils import struct_from_dict


class Trajectory(NamedTuple):
    obs: chex.ArrayTree
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    log_prob: chex.Array
    value: chex.Array
    info: dict


@struct.dataclass
class PPOHyperparams:
    gamma: float = 0.995
    gae_lambda: float = 0.98
    clip_eps: float = 0.2
    entropy_coeff: float = 1e-3
    critic_coeff: float = 0.5


@struct.dataclass
class TrainLoopShape:
    n_train_envs: int
    n_steps: int
    n_minibatches: int
    n_ppo_epochs: int
    n_updates: int
    eval_freq: int


@struct.dataclass
class OptimizerConfig:
    lr: float
    max_grad_norm: float


@struct.dataclass
class EvalConfig:
    n_eval_attempts: int
    eval_levels: Tuple[str, ...] = struct.field(pytree_node=False)


@struct.dataclass
class CheckpointConfig:
    run_name: str = struct.field(pytree_node=False)
    seed: int
    checkpoint_save_interval: int
    max_number_of_checkpoints: int


@struct.dataclass
class NetworkConfig:
    conv_filters: int = 16
    direction_embed_dim: int = 5
    hidden_dim: int = 32
    lstm_features: int = 256


class PPOUpdateBatch(NamedTuple):
    obs: chex.ArrayTree
    action: chex.Array
    done: chex.Array
    log_prob: chex.Array
    value: chex.Array
    targets: chex.Array
    advantages: chex.Array


class RolloutState(NamedTuple):
    hstate: chex.ArrayTree
    obs: Observation
    env_state: EnvState
    done: chex.Array


class TrainState(BaseTrainState):
    update_count: int
    last_hstate: chex.ArrayTree
    last_obs: chex.ArrayTree
    last_env_state: chex.ArrayTree


class ActorCritic(nn.Module):
    """Actor-critic with LSTM."""
    action_dim: Sequence[int]
    network_config: NetworkConfig

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        img_embed = nn.Conv(
            self.network_config.conv_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
        )(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(
            self.network_config.direction_embed_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="scalar_embed",
        )(dir_embed)

        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        hidden, embedding = ResetRNN(
            nn.OptimizedLSTMCell(features=self.network_config.lstm_features)
        )((embedding, dones), initial_carry=hidden)

        actor_mean = nn.Dense(
            self.network_config.hidden_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="actor0",
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor1",
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.network_config.hidden_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="critic0",
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic1",
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims, lstm_features: int):
        return nn.OptimizedLSTMCell(features=lstm_features).initialize_carry(
            jax.random.PRNGKey(0),
            (*batch_dims, lstm_features),
        )


def compute_gae(
    hparams: PPOHyperparams,
    last_value: chex.Array,
    traj: Trajectory,
) -> Tuple[chex.Array, chex.Array]:
    """Compute GAE advantages and targets from a trajectory.

    Args:
        hparams: PPO hyperparameters (uses gamma, gae_lambda)
        last_value: Shape (NUM_ENVS,)
        traj: Trajectory with shape (NUM_STEPS, NUM_ENVS, ...)

    Returns:
        (advantages, targets), each of shape (NUM_STEPS, NUM_ENVS)
    """
    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + hparams.gamma * next_value * (1 - done) - value
        gae = delta + hparams.gamma * hparams.gae_lambda * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (traj.value, traj.reward, traj.done),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj.value


def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    *,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    n_envs: int,
    max_episode_length: int,
) -> Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Trajectory]:
    """Sample trajectories using the agent in train_state.

    Args:
        rng: Singleton PRNG key
        train_state: Singleton train state
        init_hstate: RNN hidden state, shape (NUM_ENVS, ...)
        init_obs: Initial observation, shape (NUM_ENVS, ...)
        init_env_state: Initial env state, shape (NUM_ENVS, ...)
        env: Environment instance
        env_params: Environment parameters
        n_envs: Number of parallel environments
        max_episode_length: Rollout length

    Returns:
        ((rng, train_state, hstate, last_obs, last_env_state, last_value), traj)
    """
    def sample_step(carry, _):
        rng, train_state, rollout_state = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda t: t[None, ...], (rollout_state.obs, rollout_state.done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, rollout_state.hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, n_envs), rollout_state.env_state, action, env_params)

        next_rollout_state = RolloutState(hstate, next_obs, env_state, done)
        carry = (rng, train_state, next_rollout_state)
        return carry, Trajectory(rollout_state.obs, action, reward, done, log_prob, value, info)

    init_rollout_state = RolloutState(
        init_hstate,
        init_obs,
        init_env_state,
        jnp.zeros(n_envs, dtype=bool),
    )
    (rng, train_state, rollout_state), traj = jax.lax.scan(
        sample_step,
        (rng, train_state, init_rollout_state),
        None,
        length=max_episode_length,
    )

    x = jax.tree_util.tree_map(lambda t: t[None, ...], (rollout_state.obs, rollout_state.done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, rollout_state.hstate)

    return (
        rng,
        train_state,
        rollout_state.hstate,
        rollout_state.obs,
        rollout_state.env_state,
        last_value.squeeze(0),
    ), traj


def evaluate_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    *,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    max_episode_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Run the RNN policy on the environment and return (states, rewards, episode_lengths).

    Args:
        rng: PRNG key
        train_state: Agent train state
        init_hstate: Shape (n_levels,)
        init_obs: Shape (n_levels,)
        init_env_state: Shape (n_levels,)
        env: Environment instance
        env_params: Environment parameters
        max_episode_length: Maximum steps

    Returns:
        (states, rewards, episode_lengths) with shapes
        (NUM_STEPS, NUM_LEVELS, ...), (NUM_STEPS, NUM_LEVELS), (NUM_LEVELS,)
    """
    first_obs_leaf = jax.tree_util.tree_leaves(init_obs)[0]
    n_levels = first_obs_leaf.shape[0]

    def step(carry, _):
        rng, rollout_state, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda t: t[None, ...], (rollout_state.obs, rollout_state.done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, rollout_state.hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, n_levels), rollout_state.env_state, action, env_params)

        next_mask = mask & ~done
        episode_length += mask

        next_rollout_state = RolloutState(hstate, obs, next_state, done)
        return (rng, next_rollout_state, next_mask, episode_length), (rollout_state.env_state, reward)

    init_rollout_state = RolloutState(
        init_hstate,
        init_obs,
        init_env_state,
        jnp.zeros(n_levels, dtype=bool),
    )
    (_, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (rng, init_rollout_state, jnp.ones(n_levels, dtype=bool), jnp.zeros(n_levels, dtype=jnp.int32)),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths


def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: PPOUpdateBatch,
    hparams: PPOHyperparams,
    *,
    n_envs: int,
    n_steps: int,
    n_minibatches: int,
    n_ppo_epochs: int,
    update_grad: bool = True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """Update the actor-critic using PPO on a batch of rollout data.

    Args:
        rng: PRNG key
        train_state: Current train state
        init_hstate: Initial RNN hidden state
        batch: PPO update batch
        hparams: PPO hyperparameters (clip_eps, entropy_coeff, critic_coeff)
        n_envs: Number of environments
        n_steps: Number of rollout steps
        n_minibatches: Number of minibatches
        n_ppo_epochs: Number of PPO epochs
        update_grad: If False, skip applying gradients

    Returns:
        ((rng, train_state), losses) where losses = (loss, (l_vf, l_clip, entropy))
    """
    last_done = jnp.roll(batch.done, 1, axis=0).at[0].set(False)
    batch = batch._replace(done=last_done)

    clip_eps = hparams.clip_eps
    entropy_coeff = hparams.entropy_coeff
    critic_coeff = hparams.critic_coeff

    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, action, done, log_prob, value, targets, advantages = minibatch

            def loss_fn(params):
                _, pi, value_pred = train_state.apply_fn(params, (obs, done), init_hstate)
                log_prob_pred = pi.log_prob(action)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_prob_pred - log_prob)
                normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                unclipped_objective = ratio * normalized_advantages
                clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                clipped_objective = clipped_ratio * normalized_advantages
                l_clip = -jnp.minimum(unclipped_objective, clipped_objective).mean()

                value_pred_clipped = value + (value_pred - value).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum((value_pred - targets) ** 2, (value_pred_clipped - targets) ** 2).mean()

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy

                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, n_envs)

        shuffled_init_hstate = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0),
            init_hstate,
        )
        minibatched_init_hstate = jax.tree_util.tree_map(
            lambda x: x.reshape(n_minibatches, -1, *x.shape[1:]),
            shuffled_init_hstate,
        )

        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=1),
            batch,
        )
        reshaped_batch = jax.tree_util.tree_map(
            lambda x: x.reshape(x.shape[0], n_minibatches, -1, *x.shape[2:]),
            shuffled_batch,
        )
        minibatched_batch = jax.tree_util.tree_map(
            lambda x: x.swapaxes(0, 1),
            reshaped_batch,
        )

        minibatches = (
            minibatched_init_hstate,
            *minibatched_batch,
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_ppo_epochs)


def setup_checkpointing(
    checkpoint_config: CheckpointConfig,
) -> ocp.CheckpointManager:
    """Set up orbax checkpoint manager.

    Args:
        checkpoint_config: Checkpointing settings

    Returns:
        Configured CheckpointManager
    """
    save_dir = os.path.join(
        os.getcwd(),
        "checkpoints",
        checkpoint_config.run_name,
        str(checkpoint_config.seed),
    )
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(save_dir, 'models'),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=checkpoint_config.checkpoint_save_interval,
            max_to_keep=checkpoint_config.max_number_of_checkpoints,
        )
    )
    return checkpoint_manager


def init_wandb(*, config: dict, project: str):
    run = wandb.init(
        config=config,
        project=project,
        entity=config["entity"],
        group=config["group_name"],
        name=config["run_name"],
        tags=["DR"],
    )

    wandb.define_metric("n_updates")
    wandb.define_metric("n_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="n_updates")
    wandb.define_metric("level_sampler/*", step_metric="n_updates")
    wandb.define_metric("agent/*", step_metric="n_updates")
    wandb.define_metric("return/*", step_metric="n_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="n_updates")

    return run


def log_eval(stats, *, train_loop_shape: TrainLoopShape, eval_config: EvalConfig, env_renderer, env_params):
    """Log evaluation metrics and animations to wandb.

    Args:
        stats: Metrics dictionary from train_and_eval_step
        train_loop_shape: Training loop shape settings
        eval_config: Evaluation settings
        env_renderer: MazeRenderer for generating animation frames
        env_params: Environment parameters
    """
    print(f"Logging update: {stats['update_count']}")

    env_steps = stats["update_count"] * train_loop_shape.n_train_envs * train_loop_shape.n_steps
    general_metrics = {
        "n_updates": stats["update_count"],
        "n_env_steps": env_steps,
        "sps": env_steps / stats['time_delta'],
    }

    solve_rates = stats['eval_solve_rates']
    returns = stats["eval_returns"]
    solve_rate_metrics = {f"solve_rate/{name}": solve_rate for name, solve_rate in zip(eval_config.eval_levels, solve_rates)}
    solve_rate_summary_metrics = {"solve_rate/mean": solve_rates.mean()}
    return_metrics = {f"return/{name}": ret for name, ret in zip(eval_config.eval_levels, returns)}
    return_summary_metrics = {"return/mean": returns.mean()}
    eval_length_metrics = {"eval_ep_lengths/mean": stats['eval_ep_lengths'].mean()}

    loss, (critic_loss, actor_loss, entropy) = stats["losses"]
    agent_metrics = {
        "agent/loss": loss,
        "agent/critic_loss": critic_loss,
        "agent/actor_loss": actor_loss,
        "agent/entropy": entropy,
    }

    animation_metrics = {}
    for i, level_name in enumerate(eval_config.eval_levels):
        frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
        frames = np.array(frames[:episode_length])
        animation_metrics[f"animations/{level_name}"] = wandb.Video(frames, fps=4, format="gif")

    log_dict = {
        **general_metrics,
        **solve_rate_metrics,
        **solve_rate_summary_metrics,
        **return_metrics,
        **return_summary_metrics,
        **eval_length_metrics,
        **agent_metrics,
        **animation_metrics,
    }

    wandb.log(log_dict)


def create_train_state(
    rng: chex.PRNGKey,
    *,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    sample_random_level,
    train_loop_shape: TrainLoopShape,
    optimizer_config: OptimizerConfig,
    network_config: NetworkConfig,
) -> TrainState:
    """Create the initial TrainState with network, optimizer, and env state.

    Args:
        rng: PRNG key
        env: Wrapped environment
        env_params: Environment parameters
        sample_random_level: Level generator function
        train_loop_shape: Training loop shape settings
        optimizer_config: Optimizer settings
        network_config: Network architecture settings

    Returns:
        Initialized TrainState
    """
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (train_loop_shape.n_minibatches * train_loop_shape.n_ppo_epochs))
            / train_loop_shape.n_updates
        )
        return optimizer_config.lr * frac

    obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
    init_sequence_length = network_config.lstm_features
    obs_for_env_batch = jax.tree_util.tree_map(
        lambda t: jnp.repeat(t[None, ...], train_loop_shape.n_train_envs, axis=0),
        obs,
    )
    obs = jax.tree_util.tree_map(
        lambda t: jnp.repeat(t[None, ...], init_sequence_length, axis=0),
        obs_for_env_batch,
    )
    init_x = (obs, jnp.zeros((init_sequence_length, train_loop_shape.n_train_envs)))
    network = ActorCritic(env.action_space(env_params).n, network_config=network_config)
    rng, _rng = jax.random.split(rng)
    network_params = network.init(
        _rng,
        init_x,
        ActorCritic.initialize_carry(
            (train_loop_shape.n_train_envs,),
            network_config.lstm_features,
        ),
    )
    tx = optax.chain(
        optax.clip_by_global_norm(optimizer_config.max_grad_norm),
        optax.adam(learning_rate=linear_schedule, eps=1e-5),
    )

    rng_levels, rng_reset = jax.random.split(rng)
    new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, train_loop_shape.n_train_envs))
    reset_keys = jax.random.split(rng_reset, train_loop_shape.n_train_envs)
    init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        reset_keys,
        new_levels,
        env_params,
    )

    return TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
        update_count=0,
        last_hstate=ActorCritic.initialize_carry(
            (train_loop_shape.n_train_envs,),
            network_config.lstm_features,
        ),
        last_obs=init_obs,
        last_env_state=init_env_state,
    )


def train_step(
    carry: Tuple[chex.PRNGKey, TrainState],
    _,
    *,
    hparams: PPOHyperparams,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_loop_shape: TrainLoopShape,
):
    """One training step: sample trajectories, compute GAE, update policy.

    Args:
        carry: (rng, train_state)
        _: Unused scan input
        hparams: PPO hyperparameters
        env: Environment instance
        env_params: Environment parameters
        train_loop_shape: Training loop shape settings

    Returns:
        ((rng, train_state), metrics)
    """
    rng, train_state = carry

    (rng, train_state, hstate, last_obs, last_env_state, last_value), traj = sample_trajectories_rnn(
        rng,
        train_state,
        train_state.last_hstate,
        train_state.last_obs,
        train_state.last_env_state,
        env=env,
        env_params=env_params,
        n_envs=train_loop_shape.n_train_envs,
        max_episode_length=train_loop_shape.n_steps,
    )
    advantages, targets = compute_gae(hparams, last_value, traj)

    (rng, train_state), losses = update_actor_critic_rnn(
        rng,
        train_state,
        train_state.last_hstate,
        PPOUpdateBatch(
            obs=traj.obs,
            action=traj.action,
            done=traj.done,
            log_prob=traj.log_prob,
            value=traj.value,
            targets=targets,
            advantages=advantages,
        ),
        hparams,
        n_envs=train_loop_shape.n_train_envs,
        n_steps=train_loop_shape.n_steps,
        n_minibatches=train_loop_shape.n_minibatches,
        n_ppo_epochs=train_loop_shape.n_ppo_epochs,
        update_grad=True,
    )

    metrics = {
        "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
    }

    train_state = train_state.replace(
        update_count=train_state.update_count + 1,
        last_hstate=hstate,
        last_env_state=last_env_state,
        last_obs=last_obs,
    )
    return (rng, train_state), metrics


def eval_policy(
    rng: chex.PRNGKey,
    train_state: TrainState,
    *,
    eval_env: UnderspecifiedEnv,
    env_params: EnvParams,
    eval_levels: list,
    network_config: NetworkConfig,
):
    """Evaluate the current policy on a set of named levels.

    Args:
        rng: PRNG key
        train_state: Agent train state
        eval_env: Unwrapped evaluation environment
        env_params: Environment parameters
        eval_levels: List of level name strings
        network_config: Network architecture settings

    Returns:
        (states, cum_rewards, episode_lengths)
    """
    rng, rng_reset = jax.random.split(rng)
    levels = Level.load_prefabs(eval_levels)
    n_levels = len(eval_levels)
    reset_keys = jax.random.split(rng_reset, n_levels)
    init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
        reset_keys,
        levels,
        env_params,
    )
    states, rewards, episode_lengths = evaluate_rnn(
        rng,
        train_state,
        ActorCritic.initialize_carry((n_levels,), network_config.lstm_features),
        init_obs,
        init_env_state,
        env=eval_env,
        env_params=env_params,
        max_episode_length=env_params.max_steps_in_episode,
    )
    mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
    cum_rewards = (rewards * mask).sum(axis=0)
    return states, cum_rewards, episode_lengths


def train_and_eval_step(
    runner_state,
    _,
    *,
    train_step_fn,
    eval_policy_fn,
    env_renderer,
    env_params: EnvParams,
    train_loop_shape: TrainLoopShape,
    eval_config: EvalConfig,
):
    """Run eval_freq training steps then evaluate the policy.

    Args:
        runner_state: (rng, train_state)
        _: Unused scan input
        train_step_fn: Partially-applied train_step
        eval_policy_fn: Partially-applied eval_policy
        env_renderer: MazeRenderer for animation frames
        env_params: Environment parameters
        train_loop_shape: Training loop shape settings
        eval_config: Evaluation settings

    Returns:
        ((rng, train_state), metrics)
    """
    (rng, train_state), metrics = jax.lax.scan(train_step_fn, runner_state, None, train_loop_shape.eval_freq)

    rng, rng_eval = jax.random.split(rng)
    states, cum_rewards, episode_lengths = jax.vmap(eval_policy_fn, (0, None))(
        jax.random.split(rng_eval, eval_config.n_eval_attempts),
        train_state,
    )

    eval_solve_rates = jnp.where(cum_rewards > 0, 1., 0.).mean(axis=0)
    eval_returns = cum_rewards.mean(axis=0)

    states, episode_lengths = jax.tree_util.tree_map(lambda x: x[0], (states, episode_lengths))
    images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params)
    frames = images.transpose(0, 1, 4, 2, 3)

    metrics["update_count"] = train_state.update_count
    metrics["eval_returns"] = eval_returns
    metrics["eval_solve_rates"] = eval_solve_rates
    metrics["eval_ep_lengths"] = episode_lengths
    metrics["eval_animation"] = (frames, episode_lengths)

    return (rng, train_state), metrics


def eval_checkpoint(
    *,
    checkpoint_directory: str,
    checkpoint_to_eval: int,
    eval_config: EvalConfig,
    create_train_state_fn,
    eval_policy_fn,
):
    """Load a saved checkpoint and run evaluation.

    Saves results to a .npz file in the results/ directory.

    Args:
        checkpoint_directory: Path to checkpoint directory
        checkpoint_to_eval: Step to evaluate, or -1 for latest
        eval_config: Evaluation settings
        create_train_state_fn: JIT'd create_train_state (partial-applied)
        eval_policy_fn: Partially-applied eval_policy
    """
    rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))

    def load(rng_init, checkpoint_directory: str):
        with open(os.path.join(checkpoint_directory, 'config.json')) as f:
            loaded_config = json.load(f)
        checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_directory, 'models'), ocp.PyTreeCheckpointer())

        train_state_og: TrainState = create_train_state_fn(rng_init)
        step = checkpoint_manager.latest_step() if checkpoint_to_eval == -1 else checkpoint_to_eval

        loaded_checkpoint = checkpoint_manager.restore(step)
        params = loaded_checkpoint['params']
        train_state = train_state_og.replace(params=params)
        return train_state, loaded_config

    train_state, loaded_config = load(rng_init, checkpoint_directory)
    states, cum_rewards, episode_lengths = jax.vmap(eval_policy_fn, (0, None))(
        jax.random.split(rng_eval, eval_config.n_eval_attempts), train_state,
    )
    save_loc = checkpoint_directory.replace('checkpoints', 'results')
    os.makedirs(save_loc, exist_ok=True)
    np.savez_compressed(
        os.path.join(save_loc, 'results.npz'),
        states=np.asarray(states),
        cum_rewards=np.asarray(cum_rewards),
        episode_lengths=np.asarray(episode_lengths),
        levels=loaded_config['eval_levels'],
    )
    return states, cum_rewards, episode_lengths


def main(config=None, project="egt-pop"):
    flat_config = dict(config)
    flat_config["eval_levels"] = tuple(flat_config["eval_levels"])

    hparams = struct_from_dict(PPOHyperparams, flat_config)
    train_loop_shape = struct_from_dict(TrainLoopShape, flat_config)
    optimizer_config = struct_from_dict(OptimizerConfig, flat_config)
    eval_config = struct_from_dict(EvalConfig, flat_config)
    checkpoint_config = struct_from_dict(CheckpointConfig, flat_config)
    network_config = struct_from_dict(NetworkConfig, flat_config)

    run = init_wandb(config=flat_config, project=project)

    env = Maze(max_height=13, max_width=13, agent_view_size=flat_config["agent_view_size"], normalize_obs=True)
    eval_env = env
    sample_random_level = make_level_generator(env.max_height, env.max_width, flat_config["n_walls"])
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoResetWrapper(env, sample_random_level)
    env_params = env.default_params

    create_train_state_fn = partial(
        create_train_state,
        env=env,
        env_params=env_params,
        sample_random_level=sample_random_level,
        train_loop_shape=train_loop_shape,
        optimizer_config=optimizer_config,
        network_config=network_config,
    )
    jit_create_train_state = jax.jit(create_train_state_fn)

    bound_eval_policy = partial(
        eval_policy,
        eval_env=eval_env,
        env_params=env_params,
        eval_levels=eval_config.eval_levels,
        network_config=network_config,
    )

    bound_train_step = partial(
        train_step,
        hparams=hparams,
        env=env,
        env_params=env_params,
        train_loop_shape=train_loop_shape,
    )

    train_and_eval_step_fn = partial(
        train_and_eval_step,
        train_step_fn=bound_train_step,
        eval_policy_fn=bound_eval_policy,
        env_renderer=env_renderer,
        env_params=env_params,
        train_loop_shape=train_loop_shape,
        eval_config=eval_config,
    )
    jit_train_and_eval_step = jax.jit(train_and_eval_step_fn)

    if flat_config['mode'] == 'eval':
        return eval_checkpoint(
            checkpoint_directory=flat_config['checkpoint_directory'],
            checkpoint_to_eval=flat_config['checkpoint_to_eval'],
            eval_config=eval_config,
            create_train_state_fn=jit_create_train_state,
            eval_policy_fn=bound_eval_policy,
        )

    rng = jax.random.PRNGKey(flat_config["seed"])
    rng_init, rng_train = jax.random.split(rng)

    train_state = jit_create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    if checkpoint_config.checkpoint_save_interval > 0:
        checkpoint_manager = setup_checkpointing(checkpoint_config)
        save_dir = os.path.join(
            "checkpoints", checkpoint_config.run_name, str(checkpoint_config.seed)
        )
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(flat_config, f, indent=2)

    for eval_step in range(train_loop_shape.n_updates // train_loop_shape.eval_freq):
        start_time = time.time()
        runner_state, metrics = jit_train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        log_eval(
            metrics,
            train_loop_shape=train_loop_shape,
            eval_config=eval_config,
            env_renderer=env_renderer,
            env_params=env_params,
        )
        if checkpoint_config.checkpoint_save_interval > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()

    return runner_state[1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="egt-pop")
    parser.add_argument(
        "--entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY", "callum-lawson"),
        help="W&B entity (username or org). Defaults to WANDB_ENTITY or callum-lawson.",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    # === Train vs Eval ===
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--checkpoint_directory", type=str, default=None)
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1)
    # === CHECKPOINTING ===
    parser.add_argument("--checkpoint_save_interval", type=int, default=2)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)
    # === EVAL ===
    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--n_eval_attempts", "--eval_num_attempts", dest="n_eval_attempts", type=int, default=10)
    parser.add_argument("--eval_levels", nargs='+', default=[
        "SixteenRooms",
        "SixteenRooms2",
        "Labyrinth",
        "LabyrinthFlipped",
        "Labyrinth2",
        "StandardMaze",
        "StandardMaze2",
        "StandardMaze3",
    ])
    group = parser.add_argument_group('Training params')
    # === PPO ===
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--max_grad_norm", type=float, default=0.5)
    mut_group = group.add_mutually_exclusive_group()
    mut_group.add_argument("--n_updates", "--num_updates", dest="n_updates", type=int, default=30000)
    mut_group.add_argument("--n_env_steps", "--num_env_steps", dest="n_env_steps", type=int, default=None)
    group.add_argument("--n_steps", "--num_steps", dest="n_steps", type=int, default=256)
    group.add_argument("--n_train_envs", "--num_train_envs", dest="n_train_envs", type=int, default=32)
    group.add_argument("--n_minibatches", "--num_minibatches", dest="n_minibatches", type=int, default=1)
    group.add_argument("--gamma", type=float, default=0.995)
    group.add_argument("--n_ppo_epochs", "--epoch_ppo", dest="n_ppo_epochs", type=int, default=5)
    group.add_argument("--clip_eps", type=float, default=0.2)
    group.add_argument("--gae_lambda", type=float, default=0.98)
    group.add_argument("--entropy_coeff", type=float, default=1e-3)
    group.add_argument("--critic_coeff", type=float, default=0.5)
    # === ENV CONFIG ===
    group.add_argument("--agent_view_size", type=int, default=5)
    # === DR CONFIG ===
    group.add_argument("--n_walls", type=int, default=25)

    from config_utils import load_config
    config = load_config(parser)

    # Backward compatibility for YAML files that still use legacy v1 keys.
    legacy_to_new = {
        "eval_num_attempts": "n_eval_attempts",
        "num_updates": "n_updates",
        "num_env_steps": "n_env_steps",
        "num_steps": "n_steps",
        "num_train_envs": "n_train_envs",
        "num_minibatches": "n_minibatches",
        "epoch_ppo": "n_ppo_epochs",
    }
    for legacy_key, new_key in legacy_to_new.items():
        if legacy_key in config and config.get(new_key) == parser.get_default(new_key):
            config[new_key] = config[legacy_key]

    if config["n_env_steps"] is not None:
        config["n_updates"] = config["n_env_steps"] // (config["n_train_envs"] * config["n_steps"])

    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'

    main(config, project=config["project"])
