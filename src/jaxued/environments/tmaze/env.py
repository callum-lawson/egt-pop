from typing import Tuple
import jax
import jax.numpy as jnp
import chex
from ..maze.env import (
    Maze,
    EnvState,
    EnvParams,
    OBJECT_TO_INDEX,
    make_maze_map,
)
from ..maze.level import Level
from .level import LEVEL_GOAL_LEFT, LEVEL_GOAL_RIGHT


def make_level_generator():
    """Returns a level generator that samples uniformly between
    LEVEL_GOAL_LEFT and LEVEL_GOAL_RIGHT.
    """
    def sample_level(rng: chex.PRNGKey) -> Level:
        goal_is_on_left = jax.random.uniform(rng) < 0.5
        return jax.lax.cond(
            goal_is_on_left,
            lambda: LEVEL_GOAL_LEFT,
            lambda: LEVEL_GOAL_RIGHT,
        )
    return sample_level


class TMaze(Maze):
    """T-maze for one-shot decision tasks.

    Agent at T-junction chooses left or right. Goal is hidden from observation.
    Episode terminates after first movement (forward action that changes position).

    Args:
        reward_for_goal: Reward when agent moves to goal position.
        reward_for_no_goal: Reward when agent moves but not to goal.
        **kwargs: Passed to parent Maze class.
    """

    def __init__(
        self,
        reward_for_goal: float = 1.0,
        reward_for_no_goal: float = 0.0,
        **kwargs,
    ):
        # Disable time penalty since episode ends after one step
        super().__init__(penalize_time=False, **kwargs)
        self.reward_for_goal = reward_for_goal
        self.reward_for_no_goal = reward_for_no_goal

    def init_state_from_level(self, level: Level) -> EnvState:
        # Create maze map but hide the goal tile
        maze_map = make_maze_map(level, self.agent_view_size - 1)
        goal_x, goal_y = level.goal_pos
        padding = self.agent_view_size - 1
        empty_tile = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)
        maze_map = maze_map.at[goal_y + padding, goal_x + padding, :].set(empty_tile)

        return EnvState(
            agent_pos=jnp.array(level.agent_pos, dtype=jnp.uint32),
            agent_dir=jnp.array(level.agent_dir, dtype=jnp.uint8),
            goal_pos=jnp.array(level.goal_pos, dtype=jnp.uint32),
            wall_map=jnp.array(level.wall_map, dtype=jnp.bool_),
            maze_map=maze_map,
            time=0,
            terminal=False,
        )

    def _step_agent(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[EnvState, float]:
        previous_pos = state.agent_pos
        new_state, base_reward = super()._step_agent(rng, state, action, params)

        agent_moved = jnp.any(new_state.agent_pos != previous_pos)
        reached_goal = base_reward > 0

        # Terminate on first movement
        new_state = jax.lax.cond(
            agent_moved,
            lambda s: s.replace(terminal=True),
            lambda s: s,
            new_state,
        )

        # Custom reward: goal reward if reached goal, no-goal reward if moved elsewhere
        reward = jax.lax.select(
            reached_goal,
            self.reward_for_goal,
            jax.lax.select(agent_moved, self.reward_for_no_goal, 0.0),
        )
        return new_state, reward
