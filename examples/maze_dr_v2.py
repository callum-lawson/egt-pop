import os
import json
import time
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
    trajectory: Trajectory,
) -> Tuple[chex.Array, chex.Array]:
    """Compute GAE advantages and value targets from a trajectory."""
    def compute_gae_at_timestep(carry, timestep_data):
        gae, next_value = carry
        current_value, reward, done = timestep_data
        delta = reward + hparams.gamma * next_value * (1 - done) - current_value
        gae = delta + hparams.gamma * hparams.gae_lambda * (1 - done) * gae
        return (gae, current_value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (trajectory.value, trajectory.reward, trajectory.done),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + trajectory.value


def rollout_training_trajectories_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hidden_state: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    *,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_loop_shape: TrainLoopShape,
) -> Tuple[Tuple[chex.PRNGKey, TrainState, RolloutState], Trajectory]:
    """Collect PPO training trajectories and return the next rollout state."""
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

        step_keys = jax.random.split(rng_step, n_train_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_keys, rollout_state.env_state, action, env_params)

        next_rollout_state = RolloutState(next_hidden_state, next_obs, env_state, done)
        carry = (rng, train_state, next_rollout_state)
        return carry, Trajectory(rollout_state.obs, action, reward, done, log_prob, value, info)

    init_rollout_state = RolloutState(
        init_hidden_state,
        init_obs,
        init_env_state,
        jnp.zeros(n_train_envs, dtype=bool),
    )
    (rng, train_state, rollout_state), trajectory = jax.lax.scan(
        sample_step,
        (rng, train_state, init_rollout_state),
        None,
        length=n_steps,
    )

    return (rng, train_state, rollout_state), trajectory


def compute_bootstrap_value(
    train_state: TrainState,
    rollout_state: RolloutState,
) -> chex.Array:
    """Compute the bootstrap value from a rollout state's final observation."""
    policy_inputs = jax.tree.map(lambda t: t[None, ...], (rollout_state.obs, rollout_state.done))
    _, _, last_value = train_state.apply_fn(train_state.params, policy_inputs, rollout_state.hidden_state)
    return last_value.squeeze(0)


def rollout_eval_episodes_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hidden_state: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    *,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    max_episode_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Roll out evaluation episodes to produce states, rewards, and episode lengths for metrics."""
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

        step_keys = jax.random.split(rng_step, n_levels)
        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_keys, rollout_state.env_state, action, env_params)

        next_mask = mask & ~done
        episode_length += mask

        next_rollout_state = RolloutState(next_hidden_state, obs, next_state, done)
        return (rng, next_rollout_state, next_mask, episode_length), (rollout_state.env_state, reward)

    init_rollout_state = RolloutState(
        init_hidden_state,
        init_obs,
        init_env_state,
        jnp.zeros(n_levels, dtype=bool),
    )
    initial_mask = jnp.ones(n_levels, dtype=bool)
    initial_episode_length = jnp.zeros(n_levels, dtype=jnp.int32)
    (_, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (rng, init_rollout_state, initial_mask, initial_episode_length),
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
    """Compute per-level returns by masking rewards beyond each episode length."""
    valid_step_mask = jnp.arange(max_episode_length)[..., None] < episode_lengths
    return (rewards * valid_step_mask).sum(axis=0)


def build_ppo_update_batch(
    trajectory: Trajectory,
    targets: chex.Array,
    advantages: chex.Array,
) -> PPOUpdateBatch:
    """Build a PPO update batch from rollout trajectories and computed targets."""
    return PPOUpdateBatch(
        obs=trajectory.obs,
        action=trajectory.action,
        done=trajectory.done,
        log_prob=trajectory.log_prob,
        value=trajectory.value,
        targets=targets,
        advantages=advantages,
    )


def build_rnn_minibatches(
    init_hidden_state: chex.ArrayTree,
    batch: PPOUpdateBatch,
    permutation: chex.Array,
    n_minibatches: int,
) -> Tuple[chex.ArrayTree, ...]:
    """Build shuffled rollout minibatches for recurrent PPO updates."""
    minibatched_init_hidden_state = jax.tree.map(
        lambda x: jnp.take(x, permutation, axis=0).reshape(
            n_minibatches,
            -1,
            *x.shape[1:],
        ),
        init_hidden_state,
    )

    minibatched_batch = jax.tree.map(
        lambda x: jnp.take(x, permutation, axis=1)
        .reshape(x.shape[0], n_minibatches, -1, *x.shape[2:])
        .swapaxes(0, 1),
        batch,
    )

    return (minibatched_init_hidden_state, *minibatched_batch)


def prepare_rnn_ppo_batch(batch: PPOUpdateBatch) -> PPOUpdateBatch:
    """Shift done flags to align recurrent PPO inputs with previous-step termination."""
    shifted_done = jnp.roll(batch.done, 1, axis=0).at[0].set(False)
    return batch._replace(done=shifted_done)


def compute_rnn_ppo_loss_and_grads(
    train_state: TrainState,
    minibatch: Tuple[chex.ArrayTree, ...],
    hparams: PPOHyperparams,
):
    """Compute PPO minibatch losses and gradients for the current training state."""
    init_hidden_state, obs, action, done, log_prob, value, targets, advantages = minibatch
    clip_eps = hparams.clip_eps
    entropy_coeff = hparams.entropy_coeff
    critic_coeff = hparams.critic_coeff

    def loss_fn(params):
        _, policy_distribution, value_pred = train_state.apply_fn(
            params,
            (obs, done),
            init_hidden_state,
        )
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
    return loss, grads


def run_rnn_ppo_epochs(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hidden_state: chex.ArrayTree,
    batch: PPOUpdateBatch,
    hparams: PPOHyperparams,
    *,
    train_loop_shape: TrainLoopShape,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """Run PPO epochs over RNN minibatches by computing and applying gradients."""
    n_train_envs = train_loop_shape.n_train_envs
    n_minibatches = train_loop_shape.n_minibatches
    n_ppo_epochs = train_loop_shape.n_ppo_epochs

    def update_minibatch(train_state, minibatch):
        loss, grads = compute_rnn_ppo_loss_and_grads(train_state, minibatch, hparams)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    def update_epoch(carry, _):
        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, n_train_envs)

        minibatches = build_rnn_minibatches(
            init_hidden_state,
            batch,
            permutation,
            n_minibatches,
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_ppo_epochs)


def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hidden_state: chex.ArrayTree,
    batch: PPOUpdateBatch,
    hparams: PPOHyperparams,
    *,
    train_loop_shape: TrainLoopShape,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """Run recurrent PPO actor-critic updates over rollout minibatches."""
    prepared_batch = prepare_rnn_ppo_batch(batch)
    return run_rnn_ppo_epochs(
        rng,
        train_state,
        init_hidden_state,
        prepared_batch,
        hparams,
        train_loop_shape=train_loop_shape,
    )


def setup_checkpointing(
    checkpoint_config: CheckpointConfig,
) -> ocp.CheckpointManager:
    """Create an Orbax checkpoint manager for the current run."""
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
    """Initialize Weights & Biases and register step-metric mappings."""
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
    """Parse a flat config dict into grouped runtime config dataclasses."""
    return RuntimeConfigs(
        hparams=struct_from_dict(PPOHyperparams, flat_config),
        train_loop_shape=struct_from_dict(TrainLoopShape, flat_config),
        optimizer_config=struct_from_dict(OptimizerConfig, flat_config),
        eval_config=struct_from_dict(EvalConfig, flat_config),
        checkpoint_config=struct_from_dict(CheckpointConfig, flat_config),
        network_config=struct_from_dict(NetworkConfig, flat_config),
    )


def normalize_run_config(config: dict) -> dict:
    """Normalize eval level names to a tuple in a copied run config."""
    normalized_config = dict(config)
    normalized_config["eval_levels"] = tuple(normalized_config["eval_levels"])
    return normalized_config


def create_env_components(flat_config: dict) -> EnvComponents:
    """Create maze environment runtime components from the flat config."""
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
    """Build eval solve-rate, return, and episode-length logging metrics."""
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
    """Extract agent loss components into a logging metric dictionary."""
    loss, (critic_loss, actor_loss, entropy) = stats["losses"]
    return {
        "agent/loss": loss,
        "agent/critic_loss": critic_loss,
        "agent/actor_loss": actor_loss,
        "agent/entropy": entropy,
    }


def build_animation_metrics(stats, eval_levels: Tuple[str, ...]) -> dict:
    """Build per-level GIF animation metrics for Weights & Biases logging."""
    animation_metrics = {}
    for level_index, level_name in enumerate(eval_levels):
        frames, episode_length = (
            stats["eval_animation"][0][:, level_index],
            stats["eval_animation"][1][level_index],
        )
        frames = np.array(frames[:episode_length])
        animation_metrics[f"animations/{level_name}"] = wandb.Video(frames, fps=4, format="gif")
    return animation_metrics


def evaluate_policy_attempts(
    rng: chex.PRNGKey,
    train_state: TrainState,
    *,
    eval_policy_fn,
    n_eval_attempts: int,
):
    """Evaluate policy performance across multiple stochastic evaluation attempts."""
    eval_keys = jax.random.split(rng, n_eval_attempts)
    return jax.vmap(eval_policy_fn, (0, None))(
        eval_keys,
        train_state,
    )


def render_eval_animation(
    states: chex.ArrayTree,
    episode_lengths: chex.Array,
    *,
    env_renderer,
    env_params: EnvParams,
) -> Tuple[chex.Array, chex.Array]:
    """Render evaluation states into animation frames and keep first-attempt episode lengths."""
    states, episode_lengths = jax.tree.map(lambda x: x[0], (states, episode_lengths))
    images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params)
    frames = images.transpose(0, 1, 4, 2, 3)
    return frames, episode_lengths


def log_eval(stats, *, train_loop_shape: TrainLoopShape, eval_config: EvalConfig):
    """Assemble training/eval metrics and log them to Weights & Biases."""
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
    """Build runtime callables from environment and run configs."""
    def create_train_state_fn(rng):
        return create_train_state(
            rng,
            env=env_components.env,
            env_params=env_components.env_params,
            sample_random_level=env_components.sample_random_level,
            train_loop_shape=runtime_configs.train_loop_shape,
            optimizer_config=runtime_configs.optimizer_config,
            network_config=runtime_configs.network_config,
        )

    jit_create_train_state = jax.jit(create_train_state_fn)

    def eval_policy_fn(rng, train_state):
        return eval_policy(
            rng,
            train_state,
            eval_env=env_components.eval_env,
            env_params=env_components.env_params,
            eval_levels=runtime_configs.eval_config.eval_levels,
            network_config=runtime_configs.network_config,
        )

    def train_step_fn(carry, scan_item):
        return train_step(
            carry,
            scan_item,
            hparams=runtime_configs.hparams,
            env=env_components.env,
            env_params=env_components.env_params,
            train_loop_shape=runtime_configs.train_loop_shape,
        )

    def train_and_eval_step_fn(runner_state, scan_item):
        return train_and_eval_step(
            runner_state,
            scan_item,
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
    """Build network init inputs by tiling observations and creating a done mask."""
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


def sample_level_batch(
    rng: chex.PRNGKey,
    *,
    sample_random_level,
    n_train_envs: int,
) -> chex.ArrayTree:
    """Sample a batch of random levels for parallel training environments."""
    level_keys = jax.random.split(rng, n_train_envs)
    return jax.vmap(sample_random_level)(level_keys)


def reset_envs_to_levels(
    rng: chex.PRNGKey,
    *,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    levels: chex.ArrayTree,
) -> Tuple[Observation, EnvState]:
    """Reset each parallel environment to the corresponding provided level."""
    n_train_envs = jax.tree.leaves(levels)[0].shape[0]
    reset_keys = jax.random.split(rng, n_train_envs)
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
    """Create a fresh TrainState from initialized network, optimizer, and environment carry."""
    n_train_envs = train_loop_shape.n_train_envs
    initial_hidden_state = ActorCritic.initialize_carry(
        (n_train_envs,),
        network_config.lstm_features,
    )

    rng, rng_network_init = jax.random.split(rng)
    network, network_params = initialize_actor_critic_network(
        rng_network_init,
        env=env,
        env_params=env_params,
        sample_random_level=sample_random_level,
        n_train_envs=n_train_envs,
        network_config=network_config,
        initial_hidden_state=initial_hidden_state,
    )
    optimizer = build_optimizer(
        optimizer_config=optimizer_config,
        train_loop_shape=train_loop_shape,
    )
    rng, rng_levels, rng_reset = jax.random.split(rng, 3)
    init_obs, init_env_state = initialize_training_carry_state(
        rng_levels=rng_levels,
        rng_reset=rng_reset,
        sample_random_level=sample_random_level,
        n_train_envs=n_train_envs,
        env=env,
        env_params=env_params,
    )

    return TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=optimizer,
        update_count=0,
        last_hstate=initial_hidden_state,
        last_obs=init_obs,
        last_env_state=init_env_state,
    )


def create_learning_rate_schedule(
    *,
    optimizer_config: OptimizerConfig,
    train_loop_shape: TrainLoopShape,
):
    """Create the linear learning-rate schedule used by PPO updates."""
    ppo_updates_per_step = train_loop_shape.n_minibatches * train_loop_shape.n_ppo_epochs

    def learning_rate_schedule(update_step_count):
        completed_update_steps = update_step_count // ppo_updates_per_step
        remaining_fraction = 1.0 - completed_update_steps / train_loop_shape.n_updates
        return optimizer_config.lr * remaining_fraction

    return learning_rate_schedule


def build_optimizer(
    *,
    optimizer_config: OptimizerConfig,
    train_loop_shape: TrainLoopShape,
):
    """Build the optimizer stack for actor-critic parameter updates."""
    learning_rate_schedule = create_learning_rate_schedule(
        optimizer_config=optimizer_config,
        train_loop_shape=train_loop_shape,
    )
    return optax.chain(
        optax.clip_by_global_norm(optimizer_config.max_grad_norm),
        optax.adam(learning_rate=learning_rate_schedule, eps=1e-5),
    )


def initialize_actor_critic_network(
    rng_network_init: chex.PRNGKey,
    *,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    sample_random_level,
    n_train_envs: int,
    network_config: NetworkConfig,
    initial_hidden_state: chex.ArrayTree,
):
    """Initialize the actor-critic network and return the module with initialized parameters."""
    obs, _ = env.reset_to_level(rng_network_init, sample_random_level(rng_network_init), env_params)
    init_sequence_length = network_config.lstm_features
    sequence_batched_obs, init_done = build_network_init_inputs(
        obs,
        n_train_envs=n_train_envs,
        init_sequence_length=init_sequence_length,
    )
    network_init_inputs = (sequence_batched_obs, init_done)
    network = ActorCritic(env.action_space(env_params).n, network_config=network_config)
    network_params = network.init(
        rng_network_init,
        network_init_inputs,
        initial_hidden_state,
    )
    return network, network_params


def initialize_training_carry_state(
    *,
    rng_levels: chex.PRNGKey,
    rng_reset: chex.PRNGKey,
    sample_random_level,
    n_train_envs: int,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
) -> Tuple[Observation, EnvState]:
    """Initialize the per-environment observation and state carry for training rollouts."""
    levels = sample_level_batch(
        rng_levels,
        sample_random_level=sample_random_level,
        n_train_envs=n_train_envs,
    )
    return reset_envs_to_levels(
        rng_reset,
        env=env,
        env_params=env_params,
        levels=levels,
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
    """Run one PPO training recipe from rollout collection through actor-critic update."""
    rng, train_state = carry

    (rng, train_state, rollout_state), trajectory = rollout_training_trajectories_rnn(
        rng,
        train_state,
        train_state.last_hstate,
        train_state.last_obs,
        train_state.last_env_state,
        env=env,
        env_params=env_params,
        train_loop_shape=train_loop_shape,
    )
    last_value = compute_bootstrap_value(train_state, rollout_state)
    advantages, targets = compute_gae(hparams, last_value, trajectory)
    ppo_update_batch = build_ppo_update_batch(trajectory, targets, advantages)

    (rng, train_state), losses = update_actor_critic_rnn(
        rng,
        train_state,
        train_state.last_hstate,
        ppo_update_batch,
        hparams,
        train_loop_shape=train_loop_shape,
    )

    metrics = {
        "losses": jax.tree.map(lambda x: x.mean(), losses),
    }

    train_state = train_state.replace(
        update_count=train_state.update_count + 1,
        last_hstate=rollout_state.hidden_state,
        last_env_state=rollout_state.env_state,
        last_obs=rollout_state.obs,
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
    """Evaluate the current policy on named levels with states, returns, and episode lengths."""
    rng, rng_reset = jax.random.split(rng)
    levels = Level.load_prefabs(eval_levels)
    n_levels = len(eval_levels)
    reset_keys = jax.random.split(rng_reset, n_levels)
    init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
        reset_keys,
        levels,
        env_params,
    )
    initial_eval_hidden_state = ActorCritic.initialize_carry(
        (n_levels,),
        network_config.lstm_features,
    )
    states, rewards, episode_lengths = rollout_eval_episodes_rnn(
        rng,
        train_state,
        initial_eval_hidden_state,
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
    """Run a training chunk, evaluate policy performance, and render eval animations."""
    (rng, train_state), metrics = jax.lax.scan(train_step_fn, runner_state, None, train_loop_shape.eval_freq)

    rng, rng_eval = jax.random.split(rng)
    states, episode_returns, episode_lengths = evaluate_policy_attempts(
        rng_eval,
        train_state,
        eval_policy_fn=eval_policy_fn,
        n_eval_attempts=eval_config.n_eval_attempts,
    )

    eval_solve_rates = jnp.where(episode_returns > 0, 1., 0.).mean(axis=0)
    eval_returns = episode_returns.mean(axis=0)

    frames, episode_lengths = render_eval_animation(
        states,
        episode_lengths,
        env_renderer=env_renderer,
        env_params=env_params,
    )

    eval_metrics = {
        "update_count": train_state.update_count,
        "eval_returns": eval_returns,
        "eval_solve_rates": eval_solve_rates,
        "eval_ep_lengths": episode_lengths,
        "eval_animation": (frames, episode_lengths),
    }
    combined_metrics = {**metrics, **eval_metrics}

    return (rng, train_state), combined_metrics


def load_train_state_from_checkpoint(
    *,
    rng_init: chex.PRNGKey,
    checkpoint_directory: str,
    checkpoint_to_eval: int,
    create_train_state_fn,
) -> Tuple[TrainState, dict]:
    """Load checkpointed parameters into a template train state and return them with saved config."""
    with open(os.path.join(checkpoint_directory, 'config.json')) as f:
        loaded_config = json.load(f)
    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(os.getcwd(), checkpoint_directory, 'models'),
        ocp.PyTreeCheckpointer(),
    )
    template_train_state: TrainState = create_train_state_fn(rng_init)
    step = checkpoint_manager.latest_step() if checkpoint_to_eval == -1 else checkpoint_to_eval
    loaded_checkpoint = checkpoint_manager.restore(step)
    train_state = template_train_state.replace(params=loaded_checkpoint['params'])
    return train_state, loaded_config


def save_eval_results(
    *,
    checkpoint_directory: str,
    loaded_config: dict,
    states: chex.Array,
    episode_returns: chex.Array,
    episode_lengths: chex.Array,
):
    """Save evaluation trajectories and metrics to a compressed results artifact."""
    save_loc = checkpoint_directory.replace('checkpoints', 'results')
    os.makedirs(save_loc, exist_ok=True)
    np.savez_compressed(
        os.path.join(save_loc, 'results.npz'),
        states=np.asarray(states),
        cum_rewards=np.asarray(episode_returns),
        episode_lengths=np.asarray(episode_lengths),
        levels=loaded_config['eval_levels'],
    )


def run_eval_mode(
    *,
    checkpoint_directory: str,
    checkpoint_to_eval: int,
    eval_config: EvalConfig,
    create_train_state_fn,
    eval_policy_fn,
):
    """Run evaluation from a saved checkpoint and persist the resulting metrics."""
    eval_seed_key = jax.random.PRNGKey(10000)
    rng_init, rng_eval = jax.random.split(eval_seed_key)

    train_state, loaded_config = load_train_state_from_checkpoint(
        rng_init=rng_init,
        checkpoint_directory=checkpoint_directory,
        checkpoint_to_eval=checkpoint_to_eval,
        create_train_state_fn=create_train_state_fn,
    )
    states, episode_returns, episode_lengths = evaluate_policy_attempts(
        rng_eval,
        train_state,
        eval_policy_fn=eval_policy_fn,
        n_eval_attempts=eval_config.n_eval_attempts,
    )
    save_eval_results(
        checkpoint_directory=checkpoint_directory,
        loaded_config=loaded_config,
        states=states,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
    )
    return states, episode_returns, episode_lengths


def write_checkpoint_run_config(
    *,
    checkpoint_config: CheckpointConfig,
    run_config: dict,
):
    """Write run configuration metadata into the checkpoint directory for reproducible evaluation."""
    save_dir = os.path.join("checkpoints", checkpoint_config.run_name, str(checkpoint_config.seed))
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)


def run_timed_train_and_eval_step(runner_state, *, jit_train_and_eval_step):
    """Run one train-and-eval step and attach elapsed wall-clock time to the returned metrics."""
    start_time = time.time()
    runner_state, metrics = jit_train_and_eval_step(runner_state, None)
    metrics['time_delta'] = time.time() - start_time
    return runner_state, metrics


def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    *,
    eval_step: int,
    train_state: TrainState,
):
    """Persist one training checkpoint and block until asynchronous writes are complete."""
    checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(train_state))
    checkpoint_manager.wait_until_finished()


def run_train_mode(
    *,
    run_config: dict,
    runtime_configs: RuntimeConfigs,
    runtime_functions: RuntimeFunctions,
    project: str,
):
    """Run training with periodic evaluation, logging, and optional checkpoint persistence."""
    init_wandb(config=run_config, project=project)

    rng = jax.random.PRNGKey(run_config["seed"])
    rng_init, rng_train = jax.random.split(rng)

    train_state = runtime_functions.jit_create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    checkpoint_config = runtime_configs.checkpoint_config
    if checkpoint_config.checkpoint_save_interval > 0:
        checkpoint_manager = setup_checkpointing(checkpoint_config)
        write_checkpoint_run_config(
            checkpoint_config=checkpoint_config,
            run_config=run_config,
        )

    train_loop_shape = runtime_configs.train_loop_shape
    n_eval_steps = train_loop_shape.n_updates // train_loop_shape.eval_freq
    for eval_step in range(n_eval_steps):
        runner_state, metrics = run_timed_train_and_eval_step(
            runner_state,
            jit_train_and_eval_step=runtime_functions.jit_train_and_eval_step,
        )
        _, train_state = runner_state
        log_eval(
            metrics,
            train_loop_shape=train_loop_shape,
            eval_config=runtime_configs.eval_config,
        )
        if checkpoint_config.checkpoint_save_interval > 0:
            save_checkpoint(
                checkpoint_manager,
                eval_step=eval_step,
                train_state=train_state,
            )

    return train_state


def main(config=None, project="egt-pop"):
    """Run train or eval mode from the provided config and project settings."""
    run_config = normalize_run_config(config)
    runtime_configs = parse_runtime_configs(run_config)

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
        project=project,
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
