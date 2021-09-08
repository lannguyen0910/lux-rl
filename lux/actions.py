from lux.game import *
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.game_objects import *
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

from .find_cluster import *
import re
import collections


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
        game_state.player.research_points += 1
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

    for city_tile in city_tiles[::-1]:
        if not city_tile.can_act():
            continue

        unit_limit_exceeded = (units_cnt >= units_cap)  # recompute every time

        if player.researched_uranium() and unit_limit_exceeded:
            print("skip city", city_tile.cityid,
                  city_tile.pos.x, city_tile.pos.y)
            continue

        if not player.researched_uranium() and game_state.turns_to_night < 3:
            print("research and dont build units at night",
                  city_tile.pos.x, city_tile.pos.y)
            do_research(city_tile)
            continue

        nearest_resource_distance = game_state.distance_from_resource[
            city_tile.pos.y, city_tile.pos.x]
        travel_range = game_state.turns_to_night // GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]["WORKER"]
        resource_in_travel_range = nearest_resource_distance < travel_range

        if resource_in_travel_range and not unit_limit_exceeded:
            print("build_worker", city_tile.cityid, city_tile.pos.x,
                  city_tile.pos.y, nearest_resource_distance, travel_range)
            build_workers(city_tile)
            continue

        if not player.researched_uranium():
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
    missions.cleanup(player)  # remove dead units

    unit_ids_with_missions_assigned_this_turn = set()

    for distance_threshold in [1, 2, 3, 4, 10, 22, 30, 100]:
      for unit in player.units:
        # mission is planned regardless whether the unit can act

        if unit.id in unit_ids_with_missions_assigned_this_turn:
            continue

        # avoid sharing the same target
        game_state.repopulate_targets(missions)

        # if the unit is waiting for dawn at the side of resource
        stay_up_till_dawn = (unit.get_cargo_space_left() <= 4 and (
            not game_state.is_day_time or game_state.turn % 40 == 0))
        # if the unit is full and it is going to be day the next few days
        # go to an empty tile and build a citytile
        # print(unit.id, unit.get_cargo_space_left())
        if unit.get_cargo_space_left() == 0 or stay_up_till_dawn:
            nearest_position, nearest_distance = game_state.get_nearest_empty_tile_and_distance(
                unit.pos)
            if stay_up_till_dawn or nearest_distance * 2 <= game_state.turns_to_night - 2:
                print("plan mission to build citytile",
                      unit.id, unit.pos, "->", nearest_position)
                mission = Mission(unit.id, nearest_position, unit.build_city())
                missions.add(mission)
                unit_ids_with_missions_assigned_this_turn.add(unit.id)
                continue

        if unit.id in missions:
            mission: Mission = missions[unit.id]
            if mission.target_position == unit.pos:
                # take action and not make missions if already at position
                continue

        if unit.id in missions:
            # the mission will be recaluated if the unit fails to make a move
            continue

        best_position, best_cell_value = find_best_cluster(
            game_state, unit, -0.5, DEBUG=DEBUG)
        # [TODO] what if best_cell_value is zero
        if unit.pos - best_position > distance_threshold:
            continue
        print("plan mission adaptative", unit.id,
              unit.pos, "->", best_position)
        mission = Mission(unit.id, best_position, None)
        missions.add(mission)
        unit_ids_with_missions_assigned_this_turn.add(unit.id)

        # [TODO] when you can secure a city all the way to the end of time, do it

        # [TODO] avoid overlapping missions

        # [TODO] abort mission if block for multiple turns

    return missions


def make_unit_actions(game_state: Game, missions: Missions, DEBUG=False) -> Tuple[Missions, List[str]]:
    if DEBUG:
        print = __builtin__.print
    else:
        print = lambda *args: None

    player, opponent = game_state.player, game_state.opponent
    actions = []

    units_with_mission_but_no_action = set(missions.keys())
    prev_actions_len = -1
    while prev_actions_len < len(actions):
      prev_actions_len = len(actions)

      for unit in player.units:
        if not unit.can_act():
            units_with_mission_but_no_action.discard(unit.id)
            continue

        # if there is no mission, continue
        if unit.id not in missions:
            units_with_mission_but_no_action.discard(unit.id)
            continue

        mission: Mission = missions[unit.id]

        print("attempting action for", unit.id, unit.pos)

        # if the location is reached, take action
        if unit.pos == mission.target_position:
            units_with_mission_but_no_action.discard(unit.id)
            print("location reached and make action", unit.id, unit.pos)
            action = mission.target_action
            if action:
                actions.append(action)
            del missions[unit.id]
            continue

        # the unit will need to move
        direction = attempt_direction_to(
            game_state, unit, mission.target_position)
        if direction != "c":
            units_with_mission_but_no_action.discard(unit.id)
            action = unit.move(direction)
            print("make move", unit.id, unit.pos, direction)
            actions.append(action)
            continue

        # [TODO] make it possible for units to swap positions

    for unit_id in units_with_mission_but_no_action:
        mission: Mission = missions[unit_id]
        mission.delays += 1
        if mission.delays >= 1:
            del missions[unit_id]

    return missions, actions


def calculate_path_distance(game_state: Game, start_pos: Position, target_pos: Position, ignored_set: Set):
    if start_pos == target_pos:
        return 0

    xy_to_distance = {}
    xy_to_distance[tuple(start_pos)] = 0

    d4 = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    stack = collections.deque([tuple(start_pos)])
    while stack:
        x, y = stack.popleft()
        for dx, dy in d4:
            xx, yy = x+dx, y+dy
            # or (xx,yy) in ignored_set:
            if (xx, yy) in xy_to_distance or (xx, yy) in game_state.occupied_xy_set:
                continue
            xy_to_distance[xx, yy] = xy_to_distance[x, y] + 1
            stack.append((xx, yy))

            if (xx, yy) == tuple(target_pos):
                return xy_to_distance[xx, yy]

    return 1001


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

        if tuple(newpos) in game_state.occupied_xy_set:
            continue

        # [TODO] do not go into a city tile if you are carry substantial max wood
        if tuple(newpos) in game_state.player_city_tile_xy_set and unit.cargo.wood >= 60:
            continue

        # dist = calculate_path_distance(game_state, newpos, target_pos)
        dist = calculate_path_distance(
            game_state, newpos, target_pos, game_state.player_city_tile_xy_set)

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
