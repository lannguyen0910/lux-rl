import math
r"""" Game Strategy
- Build only one city that is constantly growing
- Build new city tiles only when we have enough fuel for them for at least 2 next night (or until the end)
- Build no more than 1 city tile per step
- All units go to city before night
- Whenever having a chance, build a new worker (and never cart)
- Have a straightforward collision avoidance approach
"""

import os
import pickle

import numpy as np
import builtins as __builtin__

from lux.game import Game, Mission, Missions
import lux.annotate as annotate

from lux.actions import *
from lux.find_cluster import *
from typing import DefaultDict

game_state = Game()
missions = Missions()


def print_and_annotate_missions(game_state: Game, missions: Missions, DEBUG=False):
    if DEBUG:
        print = __builtin__.print
    else:
        print = lambda *args: None

    print("Missions")
    print(missions)
    # you can also read the pickled missions and print its attributes

    annotations: List[str] = []
    player: Player = game_state.player

    for unit_id, mission in missions.items():
        mission: Mission = mission
        unit: Unit = player.units_by_id[unit_id]

        annotation = annotate.line(
            unit.pos.x, unit.pos.y, mission.target_position.x, mission.target_position.y)
        annotations.append(annotation)

        if mission.target_action and mission.target_action.split(" ")[0] == "bcity":
            annotation = annotate.circle(
                mission.target_position.x, mission.target_position.y)
            annotations.append(annotation)
        else:
            annotation = annotate.x(
                mission.target_position.x, mission.target_position.y)
            annotations.append(annotation)

    annotation = annotate.sidetext("U:{} C:{} L:{}/{}".format(len(game_state.player.units),
                                                              len(game_state.player_city_tile_xy_set),
                                                              len(game_state.targeted_leaders),
                                                              game_state.xy_to_resource_group_id.get_group_count()))
    annotations.append(annotation)

    return annotations


def annotate_movements(game_state: Game, actions_by_units: List[str]):
    annotations = []
    dirs = [
        DIRECTIONS.NORTH,
        DIRECTIONS.EAST,
        DIRECTIONS.SOUTH,
        DIRECTIONS.WEST,
        DIRECTIONS.CENTER
    ]
    d5 = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]

    for action_by_units in actions_by_units:
        if action_by_units[:2] != "m ":
            continue
        unit_id, dir = action_by_units.split(" ")[1:]
        unit = game_state.player.units_by_id[unit_id]
        x, y = unit.pos.x, unit.pos.y
        dx, dy = d5[dirs.index(dir)]
        annotation = annotate.line(x, y, x+dx, y+dy)
        annotations.append(annotation)

    return annotations
