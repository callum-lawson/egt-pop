"""T-maze trait utilities: biased level generation and choice measurement."""

import jax
import jax.numpy as jnp
import chex
from jaxued.environments.tmaze import LEVEL_GOAL_LEFT, LEVEL_GOAL_RIGHT

RIGHT = 1.0
LEFT = 0.0


def make_level_generator(p_right: float):
    """Create level generator with configurable goal-side probability.

    Args:
        p_right: Probability that goal is on right (env_trait).

    Returns:
        Function (rng) -> (level, env_trait) where env_trait is RIGHT or LEFT.
    """
    def sample_level(rng: chex.PRNGKey):
        goal_is_on_right = jax.random.uniform(rng) < p_right
        level = jax.lax.cond(
            goal_is_on_right,
            lambda: LEVEL_GOAL_RIGHT,
            lambda: LEVEL_GOAL_LEFT,
        )
        env_trait = jax.lax.select(goal_is_on_right, RIGHT, LEFT)
        return level, env_trait
    return sample_level


def infer_choice(env_trait: float, reward: float) -> float:
    """Infer agent's choice from level type and reward.

    In T-maze: reward > 0 means agent chose correctly (reached goal).
    Combined with knowing which side the goal was on, we can infer choice.

    Args:
        env_trait: RIGHT (1.0) if goal was on right, LEFT (0.0) if on left.
        reward: Episode reward (> 0 if goal reached).

    Returns:
        RIGHT (1.0) if agent chose right, LEFT (0.0) if chose left.
    """
    correct = reward > 0
    # If correct and goal was right -> chose right
    # If correct and goal was left -> chose left
    # If wrong and goal was right -> chose left
    # If wrong and goal was left -> chose right
    return jax.lax.select(correct, env_trait, 1.0 - env_trait)


def compute_agent_trait(choices: chex.Array) -> float:
    """Compute agent trait = P(choose right) from array of choices.

    Args:
        choices: Array of RIGHT/LEFT values.

    Returns:
        Mean choice (P(choose right)). Returns NaN if array is empty.
    """
    return jnp.mean(choices)
