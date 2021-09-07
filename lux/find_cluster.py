import builtins as __builtin__
import numpy as np
import math
import random

from typing import List
from .game import Game, Player
from .game_map import Cell, RESOURCE_TYPES, Position
from .constants import Constants
from .game_constants import GAME_CONSTANTS
from .annotate import *


def find_best_cluster(game_state: Game, position: Position, distance_multiplier=-0.1, DEBUG=False):
    if DEBUG:
        print = __builtin__.print
    else:
        print = lambda *args: None

    width, height = game_state.map_width, game_state.map_height

    cooldown = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"]
    travel_range = max(1, game_state.turns_to_night // cooldown - 2)
    # [TODO] consider the resources carried as well
    # [TODO] fix bug regarding nighttime travel, but just let them die perhaps

    score_matrix_wrt_pos = [[0 for _ in range(width)] for _ in range(height)]

    best_position = position
    best_cell_value = -1
    polar_offset = random.uniform(0, math.pi)

    # design a good game's matrix
    matrix = game_state.calculate_dominance_matrix(
        game_state.convolved_rate_matrix)

    for y, row in enumerate(matrix):
        for x, scores in enumerate(row):
            # [TODO] make it smarter than random

            # add random preferences in the directions
            dx, dy = abs(position.x - x), abs(position.y - y)
            polar_factor = math.sin(math.atan2(dx, dy) + polar_offset)**2
            if game_state.turn < 30:  # not so much
                polar_factor = math.sqrt(polar_factor)

            if scores > 0:
                distance = max(1, dx + dy)
                if distance <= travel_range:
                    # encourage going far away
                    # [TODO] discourage returning to explored territory
                    # [TODO] discourage going to planned locations
                    cell_value = polar_factor * scores * distance ** distance_multiplier
                    score_matrix_wrt_pos[y][x] = int(cell_value)

                    if cell_value > best_cell_value:
                        best_cell_value = cell_value
                        best_position = Position(x, y)

    print(travel_range)
    print(np.array(score_matrix_wrt_pos))

    return best_position, best_cell_value
