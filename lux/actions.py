from lux.game import Game, Player
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.game_objects import *
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

from .find_cluster import *
import re


def pretty_print(obj, indent=1, rec=0, key=''):
    # https://stackoverflow.com/questions/51753937/python-pretty-print-nested-objects
    s_indent = ' ' * indent * rec
    items = {}
    stg = s_indent

    if key != '':
        stg += str(key) + ': '

    # Discriminate && Check if final
    if isinstance(obj, list):
        items = enumerate(obj)
    elif isinstance(obj, dict):
        items = obj.items()
    elif '__dict__' in dir(obj):
        items = obj.__dict__.items()
    if not items:
        return stg + str(obj)

    # Recurse
    stg += '(' + type(obj).__name__ + ')\n'
    for k, v in items:
        stg += pretty_print(v, indent=indent, rec=rec+1, key=k) + "\n"

    # Return without empty lines
    return re.sub(r'\n\s*\n', '\n', stg)[:-1]


def make_city_actions(game_state: Game, player: Player):
    # https://www.lux-ai.org/specs-2021#CityTiles

    actions = []

    # max number of units available
    units_cap = sum([len(x.citytiles) for x in player.cities.values()])
    units_cnt = len(player.units)  # current number of units

    def do_research(city_tile: CityTile):
        action = city_tile.research()
        actions.append(action)

    def build_workers(city_tile: CityTile):
        nonlocal units_cnt
        action = city_tile.build_worker()
        actions.append(action)
        units_cnt += 1

    cities = list(player.cities.values())
    if len(cities) > 0:
        city = cities[0]
        city.citytiles = sorted(city.citytiles,
                                key=lambda city_tile: find_best_cluster(
                                    game_state, city_tile.pos)[1],
                                reverse=True)

        for city_tile in city.citytiles:
            # recompute every time
            unit_limit_exceeded = (units_cnt >= units_cap)
            if city_tile.can_act():

                if player.researched_uranium() and unit_limit_exceeded:
                    continue

                best_position, best_cell_value = find_best_cluster(
                    game_state, city_tile.pos)
                if not unit_limit_exceeded and best_cell_value > 0:
                    print("build_workers", city_tile.pos.x,
                          city_tile.pos.y, best_cell_value)
                    build_workers(city_tile)
                    continue

                if not player.researched_uranium():
                    # [TODO] dont bother researching uranium for smaller maps
                    print("research", city_tile.pos.x, city_tile.pos.y)
                    do_research(city_tile)
                    continue

                # otherwise don't do anything
    return actions
