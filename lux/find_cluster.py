import builtins as __builtin__
import numpy as np

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
    travel_range = max(0, game_state.turns_to_night // cooldown - 2)

    score_matrix_wrt_pos = [[0 for _ in range(width)] for _ in range(height)]

    best_position = position
    best_cell_value = -1

    for y, row in enumerate(game_state.convolved_fuel_matrix):
        for x, maxpool_scores in enumerate(row):
            if maxpool_scores > 0:
                distance = max(1, abs(position.x - x) + abs(position.y - y))
                if distance <= travel_range:
                    # encourage going far away
                    # [TODO] discourage returning to explored territory
                    # [TODO] discourage going to planned locations
                    cell_value = maxpool_scores * distance ** distance_multiplier
                    score_matrix_wrt_pos[y][x] = int(cell_value)

                    if cell_value > best_cell_value:
                        best_cell_value = cell_value
                        best_position = Position(x, y)

    print(travel_range)
    print(np.array(score_matrix_wrt_pos))

    return best_position, best_cell_value
