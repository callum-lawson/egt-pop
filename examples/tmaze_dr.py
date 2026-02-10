"""Domain Randomization training script for T-maze environment with trait measurement.

Trains an agent on T-maze with configurable goal-side probability (p_right).
Evaluates performance across a sweep of p_right values and measures agent traits.

Based on maze_dr.py with T-maze specific modifications:
- Biased level generation via p_right parameter
- Sweep evaluation across multiple p_right distributions
- Agent trait measurement (P(choose right))
"""

import os
import json
import time
from typing import Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import orbax.checkpoint as ocp
import wandb
from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.linen import ResetRNN
from jaxued.environments import MazeRenderer
from jaxued.environments.tmaze import TMaze
from jaxued.environments.maze import Level
from jaxued.traits.tmaze import (
    make_level_generator as make_biased_generator,
    infer_choice,
    compute_agent_trait,
    RIGHT,
    LEFT,
)
from jaxued.utils import max_mc, positive_value_loss
from jaxued.wrappers import AutoResetWrapper
import chex

class TrainState(BaseTrainState):
    update_count: int
    last_hstate: chex.ArrayTree
    last_obs: chex.ArrayTree
    last_env_state: chex.ArrayTree

# region PPO helper functions
def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Compute GAE advantages and targets.

    Args:
        gamma: Discount factor.
        lambd: GAE lambda.
        last_value: Shape (NUM_ENVS,).
        values: Shape (NUM_STEPS, NUM_ENVS).
        rewards: Shape (NUM_STEPS, NUM_ENVS).
        dones: Shape (NUM_STEPS, NUM_ENVS).

    Returns:
        (advantages, targets), each of shape (NUM_STEPS, NUM_ENVS).
    """
    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values

def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
) -> Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]:
    """Sample trajectories using the RNN policy.

    Returns:
        ((rng, train_state, hstate, last_obs, last_env_state, last_value), traj)
        where traj is (obs, action, reward, done, log_prob, value, info).
    """
    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))
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
        return carry, (obs, action, reward, done, log_prob, value, info)

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

    x = jax.tree_util.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)

    return (rng, train_state, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj

def evaluate_rnn_with_rewards(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Run RNN policy and return (states, rewards, episode_lengths).

    Args:
        rng: Random key.
        env: Environment.
        env_params: Environment parameters.
        train_state: Training state with policy.
        init_hstate: Initial hidden state, shape (num_levels,).
        init_obs: Initial observations, shape (num_levels,).
        init_env_state: Initial env states, shape (num_levels,).
        max_episode_length: Maximum steps to run.

    Returns:
        (states, rewards, episode_lengths) with shapes
        (NUM_STEPS, NUM_LEVELS), (NUM_STEPS, NUM_LEVELS), (NUM_LEVELS,).
    """
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]

    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
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
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool=True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """Update actor-critic with PPO.

    Args:
        rng: Random key.
        train_state: Current training state.
        init_hstate: Initial hidden states.
        batch: (obs, actions, dones, log_probs, values, targets, advantages).
        num_envs: Number of environments.
        n_steps: Number of steps per trajectory.
        n_minibatch: Number of minibatches.
        n_epochs: Number of PPO epochs.
        clip_eps: PPO clip epsilon.
        entropy_coeff: Entropy bonus coefficient.
        critic_coeff: Value loss coefficient.
        update_grad: Whether to actually update parameters.

    Returns:
        ((rng, train_state), losses) where losses is (loss, (l_vf, l_clip, entropy)).
    """
    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages

    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, actions, last_dones, log_probs, values, targets, advantages = minibatch

            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(params, (obs, last_dones), init_hstate)
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)).mean()

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()

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

class ActorCritic(nn.Module):
    """Actor-critic network with LSTM."""
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
# endregion

# region checkpointing
def setup_checkpointing(config: dict, train_state: TrainState, env: UnderspecifiedEnv, env_params: EnvParams) -> ocp.CheckpointManager:
    """Set up orbax checkpoint manager and save config."""
    overall_save_dir = os.path.join(os.getcwd(), "checkpoints", f"{config['run_name']}", str(config['seed']))
    os.makedirs(overall_save_dir, exist_ok=True)

    with open(os.path.join(overall_save_dir, 'config.json'), 'w+') as f:
        f.write(json.dumps(config.as_dict(), indent=True))

    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, 'models'),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config['checkpoint_save_interval'],
            max_to_keep=config['max_number_of_checkpoints'],
        )
    )
    return checkpoint_manager
#endregion

def compute_score(config, dones, values, max_returns, advantages):
    if config['score_function'] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config['score_function'] == "pvl":
        return positive_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")

def main(config=None, project="egt-pop"):
    run = wandb.init(
        config=config,
        project=project,
        entity=config["entity"],
        group=config["group_name"],
        name=config["run_name"],
        tags=["TMaze", "DR"],
    )
    config = wandb.config

    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("trait/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")

    def log_eval(stats):
        print(f"Logging update: {stats['update_count']}")

        env_steps = stats["update_count"] * config["num_train_envs"] * config["num_steps"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats['time_delta'],
        }

        # Sweep evaluation metrics
        for p_right_eval in config["sweep_p_rights"]:
            key = f"p{p_right_eval:.2f}".replace(".", "_")
            log_dict[f"solve_rate/{key}"] = float(stats["sweep_solve_rates"][p_right_eval])
            log_dict[f"trait/{key}"] = float(stats["sweep_traits"][p_right_eval])
            log_dict[f"return/{key}"] = float(stats["sweep_returns"][p_right_eval])

        log_dict["solve_rate/mean"] = float(np.mean(list(stats["sweep_solve_rates"].values())))
        log_dict["trait/overall"] = float(stats["overall_trait"])

        # Training loss
        loss, (critic_loss, actor_loss, entropy) = stats["losses"]
        log_dict.update({
            "agent/loss": loss,
            "agent/critic_loss": critic_loss,
            "agent/actor_loss": actor_loss,
            "agent/entropy": entropy,
        })

        log_dict.update({"eval_ep_lengths/mean": stats['eval_ep_lengths_mean']})

        # Animations for fixed eval levels
        for i, level_name in enumerate(["goal_left", "goal_right"]):
            frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
            frames = np.array(frames[:episode_length])
            log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4, format="gif")})

        wandb.log(log_dict)

    # Setup the T-maze environment
    env = TMaze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    eval_env = env
    env_renderer = MazeRenderer(env, tile_size=8)

    # Create biased level generator for training
    train_level_generator = make_biased_generator(config["p_right"])

    def sample_random_level(rng):
        level, _ = train_level_generator(rng)
        return level

    env = AutoResetWrapper(env, sample_random_level)
    env_params = env.default_params

    @jax.jit
    def create_train_state(rng) -> TrainState:
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["num_minibatches"] * config["epoch_ppo"]))
                / config["num_updates"]
            )
            return config["lr"] * frac

        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
            obs,
        )
        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
        network = ActorCritic(env.action_space(env_params).n)
        rng, _rng = jax.random.split(rng)
        network_params = network.init(_rng, init_x, ActorCritic.initialize_carry((config["num_train_envs"],)))
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )

        rng_levels, rng_reset = jax.random.split(rng)
        new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["num_train_envs"]))
        init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), new_levels, env_params)

        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            update_count=0,
            last_hstate=ActorCritic.initialize_carry((config["num_train_envs"],)),
            last_obs=init_obs,
            last_env_state=init_env_state,
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """Sample trajectories and update policy."""
        rng, train_state = carry

        (
            (rng, train_state, hstate, last_obs, last_env_state, last_value),
            (obs, actions, rewards, dones, log_probs, values, info),
        ) = sample_trajectories_rnn(
            rng,
            env,
            env_params,
            train_state,
            train_state.last_hstate,
            train_state.last_obs,
            train_state.last_env_state,
            config["num_train_envs"],
            config["num_steps"],
        )
        advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)

        (rng, train_state), losses = update_actor_critic_rnn(
            rng,
            train_state,
            train_state.last_hstate,
            (obs, actions, dones, log_probs, values, targets, advantages),
            config["num_train_envs"],
            config["num_steps"],
            config["num_minibatches"],
            config["epoch_ppo"],
            config["clip_eps"],
            config["entropy_coeff"],
            config["critic_coeff"],
            update_grad=True,
        )

        metrics = {
            "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
        }

        train_state = train_state.replace(
            update_count=train_state.update_count + 1,
            last_hstate=hstate,
            last_env_state=last_env_state,
            last_obs=last_obs
        )
        return (rng, train_state), metrics

    def eval_on_distribution(rng: chex.PRNGKey, train_state: TrainState, p_right_eval: float):
        """Evaluate agent on a specific p_right distribution.

        Returns:
            (solve_rate, agent_trait, mean_return, mean_ep_length)
        """
        eval_generator = make_biased_generator(p_right_eval)
        num_eval = config["eval_num_attempts"]

        # Sample levels and env_traits
        rng, rng_levels = jax.random.split(rng)
        level_rngs = jax.random.split(rng_levels, num_eval)
        levels_and_traits = jax.vmap(eval_generator)(level_rngs)
        levels, env_traits = levels_and_traits

        # Reset environments
        rng, rng_reset = jax.random.split(rng)
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, in_axes=(0, 0, None))(
            jax.random.split(rng_reset, num_eval), levels, env_params
        )

        # Run episodes
        _, rewards, episode_lengths = evaluate_rnn_with_rewards(
            rng,
            eval_env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((num_eval,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )

        # Compute cumulative rewards
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)

        # Infer choices and compute metrics
        choices = jax.vmap(infer_choice)(env_traits, cum_rewards)
        agent_trait = compute_agent_trait(choices)
        solve_rate = jnp.mean(cum_rewards > 0)
        mean_return = jnp.mean(cum_rewards)
        mean_ep_length = jnp.mean(episode_lengths)

        return solve_rate, agent_trait, mean_return, mean_ep_length, choices

    def eval_fixed_levels(rng: chex.PRNGKey, train_state: TrainState):
        """Evaluate on fixed goal_left and goal_right levels for animations."""
        from jaxued.environments.tmaze import LEVEL_GOAL_LEFT, LEVEL_GOAL_RIGHT

        levels = Level.stack([LEVEL_GOAL_LEFT, LEVEL_GOAL_RIGHT])
        num_levels = 2

        rng, rng_reset = jax.random.split(rng)
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset, num_levels), levels, env_params
        )

        states, rewards, episode_lengths = evaluate_rnn_with_rewards(
            rng,
            eval_env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )

        return states, episode_lengths

    @jax.jit
    def train_and_eval_step(runner_state, _):
        """Run training steps then evaluate across sweep."""
        # Train
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])

        # Eval on fixed levels for animations
        rng, rng_fixed = jax.random.split(rng)
        states, episode_lengths = eval_fixed_levels(rng_fixed, train_state)
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params)
        frames = images.transpose(0, 1, 4, 2, 3)

        metrics["update_count"] = train_state.update_count
        metrics["eval_animation"] = (frames, episode_lengths)

        return (rng, train_state), metrics

    def eval_sweep(rng: chex.PRNGKey, train_state: TrainState):
        """Evaluate across all p_right values in sweep."""
        sweep_solve_rates = {}
        sweep_traits = {}
        sweep_returns = {}
        all_choices = []
        all_ep_lengths = []

        for p_right_eval in config["sweep_p_rights"]:
            rng, rng_eval = jax.random.split(rng)
            solve_rate, agent_trait, mean_return, mean_ep_length, choices = eval_on_distribution(
                rng_eval, train_state, p_right_eval
            )
            sweep_solve_rates[p_right_eval] = solve_rate
            sweep_traits[p_right_eval] = agent_trait
            sweep_returns[p_right_eval] = mean_return
            all_choices.append(choices)
            all_ep_lengths.append(mean_ep_length)

        all_choices = jnp.concatenate(all_choices)
        overall_trait = compute_agent_trait(all_choices)
        eval_ep_lengths_mean = jnp.mean(jnp.array(all_ep_lengths))

        return sweep_solve_rates, sweep_traits, sweep_returns, overall_trait, eval_ep_lengths_mean

    def eval_checkpoint(og_config):
        """Evaluate a saved checkpoint."""
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))

        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, 'config.json')) as f:
                config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(
                os.path.join(os.getcwd(), checkpoint_directory, 'models'),
                item_handlers=ocp.StandardCheckpointHandler()
            )

            train_state_og: TrainState = create_train_state(rng_init)
            step = checkpoint_manager.latest_step() if og_config['checkpoint_to_eval'] == -1 else og_config['checkpoint_to_eval']

            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint['params']
            train_state = train_state_og.replace(params=params)
            return train_state, config

        train_state, config = load(rng_init, og_config['checkpoint_directory'])
        sweep_solve_rates, sweep_traits, sweep_returns, overall_trait, _ = eval_sweep(rng_eval, train_state)

        save_loc = og_config['checkpoint_directory'].replace('checkpoints', 'results')
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_loc, 'results.npz'),
            sweep_solve_rates=sweep_solve_rates,
            sweep_traits=sweep_traits,
            sweep_returns=sweep_returns,
            overall_trait=overall_trait,
            sweep_p_rights=og_config["sweep_p_rights"],
        )
        return sweep_solve_rates, sweep_traits, overall_trait

    if config['mode'] == 'eval':
        return eval_checkpoint(config)

    # Set up the train states
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)

    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)

    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time

        # Run sweep evaluation (not jitted due to dynamic loop over p_rights)
        rng_sweep = jax.random.PRNGKey(config["seed"] + eval_step)
        sweep_solve_rates, sweep_traits, sweep_returns, overall_trait, eval_ep_lengths_mean = eval_sweep(
            rng_sweep, runner_state[1]
        )
        metrics["sweep_solve_rates"] = sweep_solve_rates
        metrics["sweep_traits"] = sweep_traits
        metrics["sweep_returns"] = sweep_returns
        metrics["overall_trait"] = overall_trait
        metrics["eval_ep_lengths_mean"] = eval_ep_lengths_mean

        log_eval(metrics)

        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()

    return runner_state[1]

if __name__=="__main__":
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
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--eval_num_attempts", type=int, default=100)
    parser.add_argument("--sweep_p_rights", nargs='+', type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    group = parser.add_argument_group('Training params')
    # === PPO ===
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--max_grad_norm", type=float, default=0.5)
    mut_group = group.add_mutually_exclusive_group()
    mut_group.add_argument("--num_updates", type=int, default=2000)
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
    group.add_argument("--score_function", type=str, default="MaxMC", choices=["MaxMC", "pvl"])
    # === ENV CONFIG ===
    group.add_argument("--agent_view_size", type=int, default=5)
    # === DR CONFIG (T-maze specific) ===
    group.add_argument("--p_right", type=float, default=0.5,
                       help="Probability that training levels have goal on right (env_trait)")

    try:
        from examples.config_utils import load_config
    except ModuleNotFoundError:
        from config_utils import load_config
    config = load_config(parser)
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"])

    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'

    main(config, project=config["project"])
