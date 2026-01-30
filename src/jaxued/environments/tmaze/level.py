# T-maze levels inspired by ReMiDi (https://github.com/Michael-Beukman/ReMiDi)
#
# T-maze layout: agent starts at center of T-junction facing up,
# can turn left/right then move forward to reach goal.
# Goal is hidden at one end, directly reachable after one turn + one forward.
#
# Layout uses 13x13 grid to match maze environment defaults.
# Agent position marked with ^, goal with G, walls with #, empty with .

from ..maze.level import Level

LEVEL_GOAL_LEFT = Level.from_str("""
#############
#############
#############
#############
#############
#############
#####G^.#####
#############
#############
#############
#############
#############
#############
""")

LEVEL_GOAL_RIGHT = Level.from_str("""
#############
#############
#############
#############
#############
#############
#####.^G#####
#############
#############
#############
#############
#############
#############
""")
