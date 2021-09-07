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

    # [TODO] put logic in units
    cooldown = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"]
    travel_range = max(1, game_state.turns_to_night // cooldown - 2)

    if unit.night_turn_survivable > game_state.turns_to_dawn and not game_state.is_day_time:
        travel_range = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] // cooldown + \
            unit.night_travel_range

    if unit.night_turn_survivable > GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]:
        travel_range = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"] // cooldown + \
            unit.night_travel_range

    score_matrix_wrt_pos = game_state.init_zero_matrix()

    best_position = unit.pos
    best_cell_value = -1

    # if current cluster size has more than one agent mining
    consider_different_cluster = False
    current_leader = game_state.xy_to_resource_group_id.find(tuple(unit.pos))
    if current_leader:
        units_mining_on_current_cluster = game_state.resource_leader_to_locating_units[
            current_leader]
        if len(units_mining_on_current_cluster) > 1:
            consider_different_cluster = True

    # give very slight preference to richer matrices
    matrix = game_state.convolved_rate_matrix**0.01

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

            # if the targeted cluster is not targeted and mined
            # prefer to target the other cluster
            target_bonus = 1
            if consider_different_cluster:
                target_leader = game_state.xy_to_resource_group_id.find((x, y))
                if target_leader:
                    units_targeting_or_mining_on_target_cluster = \
                        game_state.resource_leader_to_locating_units[target_leader] | \
                        game_state.resource_leader_to_targeting_units[target_leader]

                    if len(units_targeting_or_mining_on_target_cluster) == 0:
                        target_bonus = 10

            empty_tile_bonus = 1
            if game_state.distance_from_resource[y, x] == 1:
                empty_tile_bonus = 4

            # add random preferences in the directions
            dx, dy = abs(unit.pos.x - x), abs(unit.pos.y - y)

            if matrix[y, x] > 0:
                distance = max(1, dx + dy)
                if distance <= travel_range:
                    # encourage going far away
                    # discourage returning to explored territory
                    # discourage going to planned locations
                    cell_value = target_bonus * empty_tile_bonus * \
                        matrix[y, x] * distance ** distance_multiplier
                    score_matrix_wrt_pos[y, x] = int(cell_value)

                    if cell_value > best_cell_value:
                        best_cell_value = cell_value
                        best_position = Position(x, y)

    # print(travel_range)
    # print(score_matrix_wrt_pos)

    return best_position, best_cell_value
