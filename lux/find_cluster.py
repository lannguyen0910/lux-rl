import builtins as __builtin__
import numpy as np
import math
import random

from typing import List
from .game import Game, Player
from .game_map import Cell, RESOURCE_TYPES, Unit, Position
from .constants import Constants
from .game_constants import GAME_CONSTANTS
from .annotate import *


def find_best_cluster(game_state: Game, unit: Unit, distance_multiplier=-0.1, DEBUG=False):
    if DEBUG:
        print = __builtin__.print
    else:
        print = lambda *args: None

    cooldown = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"]
    travel_range = max(1, game_state.turns_to_night // cooldown - 2)

    if unit.night_turn_survivable > game_state.turns_to_dawn and not game_state.is_day_time:
        travel_range = game_state.map_width + game_state.map_height

    if unit.night_turn_survivable > GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]:
        travel_range = game_state.map_width + game_state.map_height

    # [TODO] consider the resources carried as well
    # [TODO] fix bug regarding nighttime travel, but just let them die perhaps

    score_matrix_wrt_pos = game_state.init_zero_matrix()

    best_position = unit.pos
    best_cell_value = -1

    # design a good game's matrix
    matrix = game_state.calculate_dominance_matrix(
        game_state.convolved_rate_matrix)

    for y in range(game_state.map_height):
        for x in range(game_state.map_width):
            if (x, y) in game_state.targeted_xy_set:
                continue
            if (x, y) in game_state.opponent_city_tile_xy_set:
                continue
            if (x, y) in game_state.player_city_tile_xy_set:
                continue
            if (x, y) == tuple(unit.pos):
                continue

            # [TODO] make it smarter than random
            target_bonus = 1.5
            if distance_multiplier > 0:
                if game_state.xy_to_resource_group_id.find((x, y),) in game_state.targeted_leaders:
                    target_bonus = 1

            empty_tile_bonus = 1
            if game_state.distance_from_resource[y, x] == 1:
                empty_tile_bonus = 4

            # add random preferences in the directions
            dx, dy = abs(unit.pos.x - x), abs(unit.pos.y - y)

            if matrix[y, x] > 0:
                distance = max(1, dx + dy)
                if distance <= travel_range:
                    # encourage going far away
                    # [TODO] discourage returning to explored territory
                    # [TODO] discourage going to planned locations
                    cell_value = target_bonus * empty_tile_bonus * \
                        matrix[y, x] * distance ** distance_multiplier
                    score_matrix_wrt_pos[y, x] = int(cell_value)

                    if cell_value > best_cell_value:
                        best_cell_value = cell_value
                        best_position = Position(x, y)

    print(travel_range)
    print(score_matrix_wrt_pos)

    return best_position, best_cell_value
