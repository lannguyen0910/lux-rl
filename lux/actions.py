from lux.game import *
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


DIRECTIONS = Constants.DIRECTIONS


def make_city_actions(game_state: Game, DEBUG=False) -> List[str]:
    if DEBUG:
        print = __builtin__.print
    else:
        print = lambda *args: None

    player = game_state.player

    units_cap = sum([len(x.citytiles) for x in player.cities.values()])
    units_cnt = len(player.units)  # current number of units

    actions: List[str] = []

    def do_research(city_tile: CityTile):
        action = city_tile.research()
        actions.append(action)

    def build_workers(city_tile: CityTile):
        nonlocal units_cnt
        action = city_tile.build_worker()
        actions.append(action)
        units_cnt += 1

    city_tiles: List[CityTile] = []
    for city in player.cities.values():
        for city_tile in city.citytiles:
            city_tiles.append(city_tile)
    if not city_tiles:
        return []

    city_tiles = sorted(city_tiles,
                        key=lambda city_tile: find_best_cluster(
                            game_state, city_tile.pos)[1],
                        reverse=True)

    for city_tile in city_tiles:
        if not city_tile.can_act():
            continue

        unit_limit_exceeded = (units_cnt >= units_cap)  # recompute every time

        if player.researched_uranium() and unit_limit_exceeded:
            continue

        if not player.researched_uranium() and game_state.turns_to_night < 6:
            print("research and dont build units at night",
                  city_tile.pos.x, city_tile.pos.y)
            do_research(city_tile)
            continue

        if not player.researched_coal() and len(city_tiles) > 4 and len(city_tiles) % 2:
            # accelerate coal reasearch
            print("research for coal", city_tile.pos.x, city_tile.pos.y)
            do_research(city_tile)
            continue

        best_position, best_cell_value = find_best_cluster(
            game_state, city_tile.pos)
        if not unit_limit_exceeded and best_cell_value > 0:
            print("build_worker", city_tile.cityid,
                  city_tile.pos.x, city_tile.pos.y, best_cell_value)
            build_workers(city_tile)
            continue

        if not player.researched_uranium():
            # [TODO] dont bother researching uranium for smaller maps
            print("research", city_tile.pos.x, city_tile.pos.y)
            do_research(city_tile)
            continue

        # otherwise don't do anything

    return actions

def make_unit_missions(game_state: Game, missions: Missions, DEBUG=False) -> Missions:
    if DEBUG:
        print = __builtin__.print
    else:
        print = lambda *args: None

    player = game_state.player

    for unit in player.units:
        if not unit.can_act():
            continue

        if unit.id in missions.target_positions and unit.pos == missions.target_positions[unit.id]:
            # do not make new missions
            continue

        # if the unit is full and it is going to be day the next few days
        # go to an empty tile and build a city
        # print(unit.id, unit.get_cargo_space_left())
        if unit.get_cargo_space_left() == 0:
            nearest_position, nearest_distance = game_state.get_nearest_empty_tile_and_distance(
                unit.pos)
            if nearest_distance < game_state.turns_to_night - 5:
                print("plan mission build city", unit.id, nearest_position)
                missions.target_positions[unit.id] = nearest_position
                missions.target_actions[unit.id] = unit.build_city()
                continue

        if unit.id in missions.target_positions:  # there is already a mission
            continue

        # continue camping
        if game_state.convolved_rate_matrix[unit.pos.y][unit.pos.x] >= 80:
            continue

        # once a unit is built (detected as having max space)
        # go to the best cluster
        if unit.get_cargo_space_left() == 100:
            best_position, best_cell_value = find_best_cluster(
                game_state, unit.pos, random.uniform(-1, -0.5))
            # [TODO] what if best_cell_value is zero
            print("plan mission for fresh grad", unit.id, best_position)
            missions.target_positions[unit.id] = best_position
            missions.target_actions[unit.id] = None
            continue

        # if a unit is not receiving any resources
        # move to a place with resources
        if game_state.convolved_fuel_matrix[unit.pos.y][unit.pos.x] < 20:
            best_position, best_cell_value = find_best_cluster(
                game_state, unit.pos, random.uniform(0.5, 1))
            # [TODO] what if best_cell_value is zero
            print("plan mission relocate for resources", unit.id, best_position)
            missions.target_positions[unit.id] = best_position
            missions.target_actions[unit.id] = unit.move("c")
            continue

        # otherwise just camp and farm resources

        # [TODO] when you can secure a city all the way to the end of time, do it

        # [TODO] avoid overlapping missions

        # [TODO] abort mission if block for multiple turns

    missions.cleanup(player)
    return missions


def make_unit_actions(game_state: Game, missions: Missions, DEBUG=False) -> Tuple[Missions, List[str]]:
    if DEBUG:
        print = __builtin__.print
    else:
        print = lambda *args: None

    player, opponent = game_state.player, game_state.opponent
    actions = []

    prev_actions_len = -1
    while prev_actions_len < len(actions):
      prev_actions_len = len(actions)

      for unit in player.units:
        if not unit.can_act():
            continue

        # if there is no mission, continue
        if (unit.id not in missions.target_positions) and (unit.id not in missions.target_actions):
            continue

        print("making action for", unit.id, unit.pos)

        # if the location is reached, take action
        if unit.pos == missions.target_positions[unit.id]:
            print("location reached and make action", unit.id, unit.pos)
            action = missions.target_actions[unit.id]
            if action:
                actions.append(action)

            # missions.target_actions[unit.id] = unit.random_move()
            missions.delete(unit.id)
            continue

        # the unit will need to move
        direction = attempt_direction_to(
            game_state, unit, missions.target_positions[unit.id])
        if direction == "c":
            continue
        action = unit.move(direction)
        print("make move", unit.id, unit.pos, direction)
        actions.append(action)

        # [TODO] make it possible for units to swap position

    return missions, actions


def attempt_direction_to(game_state: Game, unit: Unit, target_pos: Position) -> DIRECTIONS:
    check_dirs = [
        DIRECTIONS.NORTH,
        DIRECTIONS.EAST,
        DIRECTIONS.SOUTH,
        DIRECTIONS.WEST,
    ]
    random.shuffle(check_dirs)
    closest_dist = 1000
    closest_dir = DIRECTIONS.CENTER
    closest_pos = unit.pos

    for direction in check_dirs:
        newpos = unit.pos.translate(direction, 1)
        dist = target_pos.distance_to(newpos)

        if tuple(newpos) in game_state.occupied_xy_set:
            continue

        # [TODO] do not go into a city tile if you are carry substantial wood in the early game
        if tuple(newpos) in game_state.player_city_tile_xy_set and unit.cargo.wood >= min(11, game_state.turns_to_dawn)*4:
            continue

        if dist < closest_dist:
            closest_dir = direction
            closest_dist = dist
            closest_pos = newpos

    if closest_dir != DIRECTIONS.CENTER:
        game_state.occupied_xy_set.discard(tuple(unit.pos))
        if tuple(closest_pos) not in game_state.player_city_tile_xy_set:
            game_state.occupied_xy_set.add(tuple(closest_pos))
        unit.cooldown += 2

    return closest_dir
