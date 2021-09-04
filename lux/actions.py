from lux.game import Game, Player
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

from .game_cluster import *
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
    units = len(player.units)  # current number of units

    cities = list(player.cities.values())
    if len(cities) > 0:
        city = cities[0]
        created_worker = (units >= units_cap)
        for city_tile in city.citytiles[::-1]:
            if city_tile.can_act():
                if created_worker:
                    # let's do research
                    action = city_tile.research()
                    actions.append(action)
                else:
                    # let's create one more unit in the last created city tile if we can
                    action = city_tile.build_worker()
                    actions.append(action)
                    created_worker = True
    return actions
