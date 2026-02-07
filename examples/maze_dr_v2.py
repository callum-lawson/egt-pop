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
    num_train_envs: int
    num_steps: int
    num_minibatches: int
    epoch_ppo: int
    num_updates: int
    eval_freq: int


@struct.dataclass
class OptimizerConfig:
    lr: float
    max_grad_norm: float


@struct.dataclass
class EvalConfig:
    eval_num_attempts: int
    eval_levels: Tuple[str, ...] = struct.field(pytree_node=False)


@struct.dataclass
class CheckpointConfig:
    run_name: str = struct.field(pytree_node=False)
    seed: int
    checkpoint_save_interval: int
    max_number_of_checkpoints: int


class PPOUpdateBatch(NamedTuple):
    obs: chex.ArrayTree
    action: chex.Array
    done: chex.Array
    log_prob: chex.Array
    value: chex.Array
    targets: chex.Array
    advantages: chex.Array


class TrainState(BaseTrainState):
    update_count: int
    last_hstate: chex.ArrayTree
    last_obs: chex.ArrayTree
    last_env_state: chex.ArrayTree


class ActorCritic(nn.Module):
    """Actor-critic with LSTM."""
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)

        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))


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
    num_envs: int,
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
        num_envs: Number of parallel environments
        max_episode_length: Rollout length

    Returns:
        ((rng, train_state, hstate, last_obs, last_env_state, last_value), traj)
    """
    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda t: t[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, Trajectory(obs, action, reward, done, log_prob, value, info)

    (rng, train_state, hstate, last_obs, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_envs, dtype=bool),
        ),
        None,
        length=max_episode_length,
    )

    x = jax.tree_util.tree_map(lambda t: t[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)

    return (rng, train_state, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj


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
        init_hstate: Shape (num_levels,)
        init_obs: Shape (num_levels,)
        init_env_state: Shape (num_levels,)
        env: Environment instance
        env_params: Environment parameters
        max_episode_length: Maximum steps

    Returns:
        (states, rewards, episode_lengths) with shapes
        (NUM_STEPS, NUM_LEVELS, ...), (NUM_STEPS, NUM_LEVELS), (NUM_LEVELS,)
    """
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]

    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda t: t[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)

        next_mask = mask & ~done
        episode_length += mask

        return (rng, hstate, obs, next_state, done, next_mask, episode_length), (state, reward)

    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (
            rng,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels, dtype=jnp.int32),
        ),
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
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    update_grad: bool = True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """Update the actor-critic using PPO on a batch of rollout data.

    Args:
        rng: PRNG key
        train_state: Current train state
        init_hstate: Initial RNN hidden state
        batch: PPO update batch
        hparams: PPO hyperparameters (clip_eps, entropy_coeff, critic_coeff)
        num_envs: Number of environments
        n_steps: Number of rollout steps
        n_minibatch: Number of minibatches
        n_epochs: Number of PPO epochs
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
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)).mean()

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
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0)
                .reshape(n_minibatch, -1, *x.shape[1:]),
                init_hstate,
            ),
            *jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)


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

    env_steps = stats["update_count"] * train_loop_shape.num_train_envs * train_loop_shape.num_steps
    log_dict = {
        "num_updates": stats["update_count"],
        "num_env_steps": env_steps,
        "sps": env_steps / stats['time_delta'],
    }

    solve_rates = stats['eval_solve_rates']
    returns = stats["eval_returns"]
    log_dict.update({f"solve_rate/{name}": solve_rate for name, solve_rate in zip(eval_config.eval_levels, solve_rates)})
    log_dict.update({"solve_rate/mean": solve_rates.mean()})
    log_dict.update({f"return/{name}": ret for name, ret in zip(eval_config.eval_levels, returns)})
    log_dict.update({"return/mean": returns.mean()})
    log_dict.update({"eval_ep_lengths/mean": stats['eval_ep_lengths'].mean()})

    loss, (critic_loss, actor_loss, entropy) = stats["losses"]
    log_dict.update({
        "agent/loss": loss,
        "agent/critic_loss": critic_loss,
        "agent/actor_loss": actor_loss,
        "agent/entropy": entropy,
    })

    for i, level_name in enumerate(eval_config.eval_levels):
        frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
        frames = np.array(frames[:episode_length])
        log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4, format="gif")})

    wandb.log(log_dict)


def create_train_state(
    rng: chex.PRNGKey,
    *,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    sample_random_level,
    train_loop_shape: TrainLoopShape,
    optimizer_config: OptimizerConfig,
) -> TrainState:
    """Create the initial TrainState with network, optimizer, and env state.

    Args:
        rng: PRNG key
        env: Wrapped environment
        env_params: Environment parameters
        sample_random_level: Level generator function
        train_loop_shape: Training loop shape settings
        optimizer_config: Optimizer settings

    Returns:
        Initialized TrainState
    """
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (train_loop_shape.num_minibatches * train_loop_shape.epoch_ppo))
            / train_loop_shape.num_updates
        )
        return optimizer_config.lr * frac

    obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
    obs = jax.tree_util.tree_map(
        lambda t: jnp.repeat(jnp.repeat(t[None, ...], train_loop_shape.num_train_envs, axis=0)[None, ...], 256, axis=0),
        obs,
    )
    init_x = (obs, jnp.zeros((256, train_loop_shape.num_train_envs)))
    network = ActorCritic(env.action_space(env_params).n)
    rng, _rng = jax.random.split(rng)
    network_params = network.init(_rng, init_x, ActorCritic.initialize_carry((train_loop_shape.num_train_envs,)))
    tx = optax.chain(
        optax.clip_by_global_norm(optimizer_config.max_grad_norm),
        optax.adam(learning_rate=linear_schedule, eps=1e-5),
    )

    rng_levels, rng_reset = jax.random.split(rng)
    new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, train_loop_shape.num_train_envs))
    init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(rng_reset, train_loop_shape.num_train_envs),
        new_levels,
        env_params,
    )

    return TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
        update_count=0,
        last_hstate=ActorCritic.initialize_carry((train_loop_shape.num_train_envs,)),
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
        num_envs=train_loop_shape.num_train_envs,
        max_episode_length=train_loop_shape.num_steps,
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
        num_envs=train_loop_shape.num_train_envs,
        n_steps=train_loop_shape.num_steps,
        n_minibatch=train_loop_shape.num_minibatches,
        n_epochs=train_loop_shape.epoch_ppo,
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
):
    """Evaluate the current policy on a set of named levels.

    Args:
        rng: PRNG key
        train_state: Agent train state
        eval_env: Unwrapped evaluation environment
        env_params: Environment parameters
        eval_levels: List of level name strings

    Returns:
        (states, cum_rewards, episode_lengths)
    """
    rng, rng_reset = jax.random.split(rng)
    levels = Level.load_prefabs(eval_levels)
    num_levels = len(eval_levels)
    init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
    states, rewards, episode_lengths = evaluate_rnn(
        rng,
        train_state,
        ActorCritic.initialize_carry((num_levels,)),
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
        jax.random.split(rng_eval, eval_config.eval_num_attempts),
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
        jax.random.split(rng_eval, eval_config.eval_num_attempts), train_state,
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
    run = wandb.init(config=config, project=project, entity=config["entity"], group=config["group_name"], name=config["run_name"], tags=["DR"])
    wandb_config = wandb.config

    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")

    hparams = PPOHyperparams(
        gamma=wandb_config["gamma"],
        gae_lambda=wandb_config["gae_lambda"],
        clip_eps=wandb_config["clip_eps"],
        entropy_coeff=wandb_config["entropy_coeff"],
        critic_coeff=wandb_config["critic_coeff"],
    )
    train_loop_shape = TrainLoopShape(
        num_train_envs=wandb_config["num_train_envs"],
        num_steps=wandb_config["num_steps"],
        num_minibatches=wandb_config["num_minibatches"],
        epoch_ppo=wandb_config["epoch_ppo"],
        num_updates=wandb_config["num_updates"],
        eval_freq=wandb_config["eval_freq"],
    )
    optimizer_config = OptimizerConfig(
        lr=wandb_config["lr"],
        max_grad_norm=wandb_config["max_grad_norm"],
    )
    eval_config = EvalConfig(
        eval_num_attempts=wandb_config["eval_num_attempts"],
        eval_levels=tuple(wandb_config["eval_levels"]),
    )
    checkpoint_config = CheckpointConfig(
        run_name=wandb_config["run_name"],
        seed=wandb_config["seed"],
        checkpoint_save_interval=wandb_config["checkpoint_save_interval"],
        max_number_of_checkpoints=wandb_config["max_number_of_checkpoints"],
    )

    env = Maze(max_height=13, max_width=13, agent_view_size=wandb_config["agent_view_size"], normalize_obs=True)
    eval_env = env
    sample_random_level = make_level_generator(env.max_height, env.max_width, wandb_config["n_walls"])
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoResetWrapper(env, sample_random_level)
    env_params = env.default_params

    jit_create_train_state = jax.jit(partial(
        create_train_state,
        env=env,
        env_params=env_params,
        sample_random_level=sample_random_level,
        train_loop_shape=train_loop_shape,
        optimizer_config=optimizer_config,
    ))

    bound_eval_policy = partial(
        eval_policy,
        eval_env=eval_env,
        env_params=env_params,
        eval_levels=eval_config.eval_levels,
    )

    bound_train_step = partial(
        train_step,
        hparams=hparams,
        env=env,
        env_params=env_params,
        train_loop_shape=train_loop_shape,
    )

    jit_train_and_eval_step = jax.jit(partial(
        train_and_eval_step,
        train_step_fn=bound_train_step,
        eval_policy_fn=bound_eval_policy,
        env_renderer=env_renderer,
        env_params=env_params,
        train_loop_shape=train_loop_shape,
        eval_config=eval_config,
    ))

    if wandb_config['mode'] == 'eval':
        return eval_checkpoint(
            checkpoint_directory=wandb_config['checkpoint_directory'],
            checkpoint_to_eval=wandb_config['checkpoint_to_eval'],
            eval_config=eval_config,
            create_train_state_fn=jit_create_train_state,
            eval_policy_fn=bound_eval_policy,
        )

    rng = jax.random.PRNGKey(wandb_config["seed"])
    rng_init, rng_train = jax.random.split(rng)

    train_state = jit_create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    if checkpoint_config.checkpoint_save_interval > 0:
        checkpoint_manager = setup_checkpointing(checkpoint_config)
        save_dir = os.path.join(
            "checkpoints", checkpoint_config.run_name, str(checkpoint_config.seed)
        )
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(wandb_config.as_dict(), f, indent=2)

    for eval_step in range(train_loop_shape.num_updates // train_loop_shape.eval_freq):
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
    parser.add_argument("--eval_num_attempts", type=int, default=10)
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
    mut_group.add_argument("--num_updates", type=int, default=30000)
    mut_group.add_argument("--num_env_steps", type=int, default=None)
    group.add_argument("--num_steps", type=int, default=256)
    group.add_argument("--num_train_envs", type=int, default=32)
    group.add_argument("--num_minibatches", type=int, default=1)
    group.add_argument("--gamma", type=float, default=0.995)
    group.add_argument("--epoch_ppo", type=int, default=5)
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
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"])

    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'

    main(config, project=config["project"])
