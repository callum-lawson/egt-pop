import os
import json
import time
from functools import partial
from typing import Callable, NamedTuple, Sequence, Tuple

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
try:
    from examples.config_utils import struct_from_dict
except ModuleNotFoundError:
    from config_utils import struct_from_dict

WANDB_STEP_METRIC = "n_updates"
WANDB_METRIC_PATTERNS = (
    "solve_rate/*",
    "level_sampler/*",
    "agent/*",
    "return/*",
    "eval_ep_lengths/*",
)


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
    hidden_state: chex.ArrayTree
    obs: Observation
    env_state: EnvState
    done: chex.Array

class RuntimeConfigs(NamedTuple):
    hparams: PPOHyperparams
    train_loop_shape: TrainLoopShape
    optimizer_config: OptimizerConfig
    eval_config: EvalConfig
    checkpoint_config: CheckpointConfig
    network_config: NetworkConfig


class RuntimeFunctions(NamedTuple):
    jit_create_train_state: Callable
    eval_policy_fn: Callable
    jit_train_and_eval_step: Callable


class EnvComponents(NamedTuple):
    env: UnderspecifiedEnv
    eval_env: UnderspecifiedEnv
    sample_random_level: Callable[[chex.PRNGKey], Level]
    env_renderer: MazeRenderer
    env_params: EnvParams


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
    def __call__(self, inputs, hidden_state):
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

        hidden_state, embedding = ResetRNN(
            nn.OptimizedLSTMCell(features=self.network_config.lstm_features)
        )((embedding, dones), initial_carry=hidden_state)

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
        policy_distribution = distrax.Categorical(logits=actor_mean)

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

        return hidden_state, policy_distribution, jnp.squeeze(critic, axis=-1)

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
    def compute_gae_at_timestep(carry, timestep_data):
        gae, next_value = carry
        current_value, reward, done = timestep_data
        delta = reward + hparams.gamma * next_value * (1 - done) - current_value
        gae = delta + hparams.gamma * hparams.gae_lambda * (1 - done) * gae
        return (gae, current_value), gae

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
    train_loop_shape: TrainLoopShape,
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
        train_loop_shape: Training loop dimensions

    Returns:
        ((rng, train_state, hidden_state, last_obs, last_env_state, last_value), traj)
    """
    n_train_envs = train_loop_shape.n_train_envs
    n_steps = train_loop_shape.n_steps

    def sample_step(carry, _):
        rng, train_state, rollout_state = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        policy_inputs = jax.tree.map(lambda t: t[None, ...], (rollout_state.obs, rollout_state.done))
        next_hidden_state, policy_distribution, value = train_state.apply_fn(
            train_state.params,
            policy_inputs,
            rollout_state.hidden_state,
        )
        action = policy_distribution.sample(seed=rng_action)
        log_prob = policy_distribution.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, n_train_envs), rollout_state.env_state, action, env_params)

        next_rollout_state = RolloutState(next_hidden_state, next_obs, env_state, done)
        carry = (rng, train_state, next_rollout_state)
        return carry, Trajectory(rollout_state.obs, action, reward, done, log_prob, value, info)

    init_rollout_state = RolloutState(
        init_hstate,
        init_obs,
        init_env_state,
        jnp.zeros(n_train_envs, dtype=bool),
    )
    (rng, train_state, rollout_state), traj = jax.lax.scan(
        sample_step,
        (rng, train_state, init_rollout_state),
        None,
        length=n_steps,
    )

    policy_inputs = jax.tree.map(lambda t: t[None, ...], (rollout_state.obs, rollout_state.done))
    _, _, last_value = train_state.apply_fn(train_state.params, policy_inputs, rollout_state.hidden_state)

    return (
        rng,
        train_state,
        rollout_state.hidden_state,
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
    first_obs_leaf = jax.tree.leaves(init_obs)[0]
    n_levels = first_obs_leaf.shape[0]

    def step(carry, _):
        rng, rollout_state, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        policy_inputs = jax.tree.map(lambda t: t[None, ...], (rollout_state.obs, rollout_state.done))
        next_hidden_state, policy_distribution, _ = train_state.apply_fn(
            train_state.params,
            policy_inputs,
            rollout_state.hidden_state,
        )
        action = policy_distribution.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, n_levels), rollout_state.env_state, action, env_params)

        next_mask = mask & ~done
        episode_length += mask

        next_rollout_state = RolloutState(next_hidden_state, obs, next_state, done)
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


def compute_episode_returns(
    rewards: chex.Array,
    episode_lengths: chex.Array,
    *,
    max_episode_length: int,
) -> chex.Array:
    valid_step_mask = jnp.arange(max_episode_length)[..., None] < episode_lengths
    return (rewards * valid_step_mask).sum(axis=0)


def build_rnn_minibatches(
    init_hstate: chex.ArrayTree,
    batch: PPOUpdateBatch,
    permutation: chex.Array,
    n_minibatches: int,
) -> Tuple[chex.ArrayTree, ...]:
    minibatched_init_hstate = jax.tree.map(
        lambda x: jnp.take(x, permutation, axis=0).reshape(
            n_minibatches,
            -1,
            *x.shape[1:],
        ),
        init_hstate,
    )

    minibatched_batch = jax.tree.map(
        lambda x: jnp.take(x, permutation, axis=1)
        .reshape(x.shape[0], n_minibatches, -1, *x.shape[2:])
        .swapaxes(0, 1),
        batch,
    )

    return (minibatched_init_hstate, *minibatched_batch)


def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: PPOUpdateBatch,
    hparams: PPOHyperparams,
    *,
    train_loop_shape: TrainLoopShape,
    update_grad: bool = True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """Update the actor-critic using PPO on a batch of rollout data.

    Args:
        rng: PRNG key
        train_state: Current train state
        init_hstate: Initial RNN hidden state
        batch: PPO update batch
        hparams: PPO hyperparameters (clip_eps, entropy_coeff, critic_coeff)
        train_loop_shape: Training loop dimensions
        update_grad: If False, skip applying gradients

    Returns:
        ((rng, train_state), losses) where losses = (loss, (value_loss, policy_loss, entropy))
    """
    last_done = jnp.roll(batch.done, 1, axis=0).at[0].set(False)
    batch = batch._replace(done=last_done)

    clip_eps = hparams.clip_eps
    entropy_coeff = hparams.entropy_coeff
    critic_coeff = hparams.critic_coeff
    n_train_envs = train_loop_shape.n_train_envs
    n_minibatches = train_loop_shape.n_minibatches
    n_ppo_epochs = train_loop_shape.n_ppo_epochs

    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, action, done, log_prob, value, targets, advantages = minibatch

            def loss_fn(params):
                _, policy_distribution, value_pred = train_state.apply_fn(params, (obs, done), init_hstate)
                log_prob_pred = policy_distribution.log_prob(action)
                entropy = policy_distribution.entropy().mean()

                ratio = jnp.exp(log_prob_pred - log_prob)
                normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                unclipped_objective = ratio * normalized_advantages
                clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
                clipped_objective = clipped_ratio * normalized_advantages
                policy_loss = -jnp.minimum(unclipped_objective, clipped_objective).mean()

                value_pred_clipped = value + (value_pred - value).clip(-clip_eps, clip_eps)
                value_loss = 0.5 * jnp.maximum(
                    (value_pred - targets) ** 2,
                    (value_pred_clipped - targets) ** 2,
                ).mean()

                loss = policy_loss + critic_coeff * value_loss - entropy_coeff * entropy

                return loss, (value_loss, policy_loss, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, n_train_envs)

        minibatches = build_rnn_minibatches(
            init_hstate,
            batch,
            permutation,
            n_minibatches,
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
    wandb.init(
        config=config,
        project=project,
        entity=config["entity"],
        group=config["group_name"],
        name=config["run_name"],
        tags=["DR"],
    )

    wandb.define_metric(WANDB_STEP_METRIC)
    wandb.define_metric("n_env_steps")
    for metric_pattern in WANDB_METRIC_PATTERNS:
        wandb.define_metric(metric_pattern, step_metric=WANDB_STEP_METRIC)


def parse_runtime_configs(flat_config: dict) -> RuntimeConfigs:
    return RuntimeConfigs(
        hparams=struct_from_dict(PPOHyperparams, flat_config),
        train_loop_shape=struct_from_dict(TrainLoopShape, flat_config),
        optimizer_config=struct_from_dict(OptimizerConfig, flat_config),
        eval_config=struct_from_dict(EvalConfig, flat_config),
        checkpoint_config=struct_from_dict(CheckpointConfig, flat_config),
        network_config=struct_from_dict(NetworkConfig, flat_config),
    )


def normalize_run_config(config: dict) -> dict:
    normalized_config = dict(config)
    normalized_config["eval_levels"] = tuple(normalized_config["eval_levels"])
    return normalized_config


def create_env_components(flat_config: dict) -> EnvComponents:
    env = Maze(
        max_height=13,
        max_width=13,
        agent_view_size=flat_config["agent_view_size"],
        normalize_obs=True,
    )
    eval_env = env
    sample_random_level = make_level_generator(
        env.max_height,
        env.max_width,
        flat_config["n_walls"],
    )
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoResetWrapper(env, sample_random_level)
    env_params = env.default_params
    return EnvComponents(
        env=env,
        eval_env=eval_env,
        sample_random_level=sample_random_level,
        env_renderer=env_renderer,
        env_params=env_params,
    )


def build_eval_metrics(stats, eval_levels: Tuple[str, ...]) -> dict:
    solve_rates = stats['eval_solve_rates']
    returns = stats["eval_returns"]
    solve_rate_metrics = {
        f"solve_rate/{name}": solve_rate
        for name, solve_rate in zip(eval_levels, solve_rates)
    }
    return_metrics = {
        f"return/{name}": ret
        for name, ret in zip(eval_levels, returns)
    }
    return {
        **solve_rate_metrics,
        "solve_rate/mean": solve_rates.mean(),
        **return_metrics,
        "return/mean": returns.mean(),
        "eval_ep_lengths/mean": stats['eval_ep_lengths'].mean(),
    }


def build_agent_metrics(stats) -> dict:
    loss, (critic_loss, actor_loss, entropy) = stats["losses"]
    return {
        "agent/loss": loss,
        "agent/critic_loss": critic_loss,
        "agent/actor_loss": actor_loss,
        "agent/entropy": entropy,
    }


def build_animation_metrics(stats, eval_levels: Tuple[str, ...]) -> dict:
    animation_metrics = {}
    for level_index, level_name in enumerate(eval_levels):
        frames, episode_length = (
            stats["eval_animation"][0][:, level_index],
            stats["eval_animation"][1][level_index],
        )
        frames = np.array(frames[:episode_length])
        animation_metrics[f"animations/{level_name}"] = wandb.Video(frames, fps=4, format="gif")
    return animation_metrics


def log_eval(stats, *, train_loop_shape: TrainLoopShape, eval_config: EvalConfig):
    """Log evaluation metrics and animations to wandb.

    Args:
        stats: Metrics dictionary from train_and_eval_step
        train_loop_shape: Training loop shape settings
        eval_config: Evaluation settings
    """
    print(f"Logging update: {stats['update_count']}")

    env_steps = stats["update_count"] * train_loop_shape.n_train_envs * train_loop_shape.n_steps
    general_metrics = {
        "n_updates": stats["update_count"],
        "n_env_steps": env_steps,
        "sps": env_steps / stats['time_delta'],
    }
    eval_metrics = build_eval_metrics(stats, eval_config.eval_levels)
    agent_metrics = build_agent_metrics(stats)
    animation_metrics = build_animation_metrics(stats, eval_config.eval_levels)

    log_dict = {
        **general_metrics,
        **eval_metrics,
        **agent_metrics,
        **animation_metrics,
    }

    wandb.log(log_dict)


def build_runtime_functions(
    *,
    env_components: EnvComponents,
    runtime_configs: RuntimeConfigs,
) -> RuntimeFunctions:
    create_train_state_fn = partial(
        create_train_state,
        env=env_components.env,
        env_params=env_components.env_params,
        sample_random_level=env_components.sample_random_level,
        train_loop_shape=runtime_configs.train_loop_shape,
        optimizer_config=runtime_configs.optimizer_config,
        network_config=runtime_configs.network_config,
    )
    jit_create_train_state = jax.jit(create_train_state_fn)

    eval_policy_fn = partial(
        eval_policy,
        eval_env=env_components.eval_env,
        env_params=env_components.env_params,
        eval_levels=runtime_configs.eval_config.eval_levels,
        network_config=runtime_configs.network_config,
    )

    train_step_fn = partial(
        train_step,
        hparams=runtime_configs.hparams,
        env=env_components.env,
        env_params=env_components.env_params,
        train_loop_shape=runtime_configs.train_loop_shape,
    )

    train_and_eval_step_fn = partial(
        train_and_eval_step,
        train_step_fn=train_step_fn,
        eval_policy_fn=eval_policy_fn,
        env_renderer=env_components.env_renderer,
        env_params=env_components.env_params,
        train_loop_shape=runtime_configs.train_loop_shape,
        eval_config=runtime_configs.eval_config,
    )
    jit_train_and_eval_step = jax.jit(train_and_eval_step_fn)

    return RuntimeFunctions(
        jit_create_train_state=jit_create_train_state,
        eval_policy_fn=eval_policy_fn,
        jit_train_and_eval_step=jit_train_and_eval_step,
    )


def build_network_init_inputs(
    obs: Observation,
    *,
    n_train_envs: int,
    init_sequence_length: int,
) -> Tuple[Observation, chex.Array]:
    env_batched_obs = jax.tree.map(
        lambda tensor: jnp.repeat(tensor[None, ...], n_train_envs, axis=0),
        obs,
    )
    sequence_batched_obs = jax.tree.map(
        lambda tensor: jnp.repeat(tensor[None, ...], init_sequence_length, axis=0),
        env_batched_obs,
    )
    init_done = jnp.zeros((init_sequence_length, n_train_envs))
    return sequence_batched_obs, init_done


def reset_env_batch(
    rng: chex.PRNGKey,
    *,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    sample_random_level,
    n_train_envs: int,
) -> Tuple[Observation, EnvState]:
    rng_levels, rng_reset = jax.random.split(rng)
    levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, n_train_envs))
    reset_keys = jax.random.split(rng_reset, n_train_envs)
    return jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        reset_keys,
        levels,
        env_params,
    )


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
    def learning_rate_schedule(update_step_count):
        frac = (
            1.0
            - (
                update_step_count
                // (train_loop_shape.n_minibatches * train_loop_shape.n_ppo_epochs)
            )
            / train_loop_shape.n_updates
        )
        return optimizer_config.lr * frac

    n_train_envs = train_loop_shape.n_train_envs
    obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
    init_sequence_length = network_config.lstm_features
    sequence_batched_obs, init_done = build_network_init_inputs(
        obs,
        n_train_envs=n_train_envs,
        init_sequence_length=init_sequence_length,
    )
    network_init_inputs = (sequence_batched_obs, init_done)
    network = ActorCritic(env.action_space(env_params).n, network_config=network_config)
    rng, rng_network_init = jax.random.split(rng)
    network_params = network.init(
        rng_network_init,
        network_init_inputs,
        ActorCritic.initialize_carry(
            (n_train_envs,),
            network_config.lstm_features,
        ),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(optimizer_config.max_grad_norm),
        optax.adam(learning_rate=learning_rate_schedule, eps=1e-5),
    )

    init_obs, init_env_state = reset_env_batch(
        rng,
        env=env,
        env_params=env_params,
        sample_random_level=sample_random_level,
        n_train_envs=n_train_envs,
    )

    return TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=optimizer,
        update_count=0,
        last_hstate=ActorCritic.initialize_carry(
            (n_train_envs,),
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

    (rng, train_state, next_hidden_state, last_obs, last_env_state, last_value), traj = sample_trajectories_rnn(
        rng,
        train_state,
        train_state.last_hstate,
        train_state.last_obs,
        train_state.last_env_state,
        env=env,
        env_params=env_params,
        train_loop_shape=train_loop_shape,
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
        train_loop_shape=train_loop_shape,
        update_grad=True,
    )

    metrics = {
        "losses": jax.tree.map(lambda x: x.mean(), losses),
    }

    train_state = train_state.replace(
        update_count=train_state.update_count + 1,
        last_hstate=next_hidden_state,
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
        (states, episode_returns, episode_lengths)
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
    episode_returns = compute_episode_returns(
        rewards,
        episode_lengths,
        max_episode_length=env_params.max_steps_in_episode,
    )
    return states, episode_returns, episode_lengths


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
    states, episode_returns, episode_lengths = jax.vmap(eval_policy_fn, (0, None))(
        jax.random.split(rng_eval, eval_config.n_eval_attempts),
        train_state,
    )

    eval_solve_rates = jnp.where(episode_returns > 0, 1., 0.).mean(axis=0)
    eval_returns = episode_returns.mean(axis=0)

    states, episode_lengths = jax.tree.map(lambda x: x[0], (states, episode_lengths))
    images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params)
    frames = images.transpose(0, 1, 4, 2, 3)

    eval_metrics = {
        "update_count": train_state.update_count,
        "eval_returns": eval_returns,
        "eval_solve_rates": eval_solve_rates,
        "eval_ep_lengths": episode_lengths,
        "eval_animation": (frames, episode_lengths),
    }
    combined_metrics = {**metrics, **eval_metrics}

    return (rng, train_state), combined_metrics


def run_eval_mode(
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

        template_train_state: TrainState = create_train_state_fn(rng_init)
        step = checkpoint_manager.latest_step() if checkpoint_to_eval == -1 else checkpoint_to_eval

        loaded_checkpoint = checkpoint_manager.restore(step)
        params = loaded_checkpoint['params']
        train_state = template_train_state.replace(params=params)
        return train_state, loaded_config

    train_state, loaded_config = load(rng_init, checkpoint_directory)
    states, episode_returns, episode_lengths = jax.vmap(eval_policy_fn, (0, None))(
        jax.random.split(rng_eval, eval_config.n_eval_attempts), train_state,
    )
    save_loc = checkpoint_directory.replace('checkpoints', 'results')
    os.makedirs(save_loc, exist_ok=True)
    np.savez_compressed(
        os.path.join(save_loc, 'results.npz'),
        states=np.asarray(states),
        cum_rewards=np.asarray(episode_returns),
        episode_lengths=np.asarray(episode_lengths),
        levels=loaded_config['eval_levels'],
    )
    return states, episode_returns, episode_lengths


def run_train_mode(
    *,
    run_config: dict,
    runtime_configs: RuntimeConfigs,
    runtime_functions: RuntimeFunctions,
):
    rng = jax.random.PRNGKey(run_config["seed"])
    rng_init, rng_train = jax.random.split(rng)

    train_state = runtime_functions.jit_create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    checkpoint_config = runtime_configs.checkpoint_config
    if checkpoint_config.checkpoint_save_interval > 0:
        checkpoint_manager = setup_checkpointing(checkpoint_config)
        save_dir = os.path.join(
            "checkpoints", checkpoint_config.run_name, str(checkpoint_config.seed)
        )
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(run_config, f, indent=2)

    train_loop_shape = runtime_configs.train_loop_shape
    for eval_step in range(train_loop_shape.n_updates // train_loop_shape.eval_freq):
        start_time = time.time()
        runner_state, metrics = runtime_functions.jit_train_and_eval_step(runner_state, None)
        end_time = time.time()
        metrics['time_delta'] = end_time - start_time
        log_eval(
            metrics,
            train_loop_shape=train_loop_shape,
            eval_config=runtime_configs.eval_config,
        )
        if checkpoint_config.checkpoint_save_interval > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()

    return runner_state[1]


def main(config=None, project="egt-pop"):
    run_config = normalize_run_config(config)
    runtime_configs = parse_runtime_configs(run_config)

    init_wandb(config=run_config, project=project)

    env_components = create_env_components(run_config)
    runtime_functions = build_runtime_functions(
        env_components=env_components,
        runtime_configs=runtime_configs,
    )

    if run_config['mode'] == 'eval':
        return run_eval_mode(
            checkpoint_directory=run_config['checkpoint_directory'],
            checkpoint_to_eval=run_config['checkpoint_to_eval'],
            eval_config=runtime_configs.eval_config,
            create_train_state_fn=runtime_functions.jit_create_train_state,
            eval_policy_fn=runtime_functions.eval_policy_fn,
        )

    return run_train_mode(
        run_config=run_config,
        runtime_configs=runtime_configs,
        runtime_functions=runtime_functions,
    )


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

    try:
        from examples.config_utils import load_config
    except ModuleNotFoundError:
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
