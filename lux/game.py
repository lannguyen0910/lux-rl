from typing import DefaultDict, Dict, List, Tuple, Set
from collections import defaultdict, deque
from .constants import Constants
from .game_map import RESOURCE_TYPES, GameMap
from .game_objects import Player, Unit, City, CityTile
from .game_position import Position
from .game_constants import GAME_CONSTANTS


INPUT_CONSTANTS = Constants.INPUT_CONSTANTS


class Mission:
    """
    Each mission is defined with a target position and optionally an action (e.g. build citytile)
    """

    def __init__(self, unit_id: str, target_position: Position, target_action: str = ""):
        self.target_position: Position = target_position
        self.target_action: str = target_action
        self.unit_id: str = unit_id
        self.delays: int = 0

    def __str__(self):
        return " ".join([str(self.target_position), self.target_action])


class Missions(defaultdict):
    """
    Plan missions for each unit in arbitrary order.
    """

    def __init__(self):
        self: DefaultDict[str, Mission] = defaultdict(Mission)

    def add(self, mission: Mission):
        self[mission.unit_id] = mission

    def cleanup(self, player: Player, player_city_tile_xy_set: Set[Tuple], opponent_city_tile_xy_set: Set[Tuple]):
        for unit_id in list(self.keys()):
            mission: Mission = self[unit_id]

            # if dead, delete from list
            if unit_id not in player.units_by_id:
                del self[unit_id]
                continue

            unit: Unit = player.units_by_id[unit_id]

            # if want to build city without resource, delete from list
            if mission.target_action and mission.target_action[:5] == "bcity":
                if unit.cargo == 0:
                    del self[unit_id]
                    continue

            # if opponent has already built a base, reconsider your mission
            if tuple(mission.target_position) in opponent_city_tile_xy_set:
                del self[unit_id]
                continue

            # if are in a base, reconsider your mission
            if tuple(unit.pos) in player_city_tile_xy_set:
                del self[unit_id]
                continue

    def __str__(self):
        return " ".join([unit_id + " " + str(x) for unit_id, x in self.items()])

    def get_targets(self):
        return [mission.target_position for unit_id, mission in self.items()]


class Game:
    def _initialize(self, messages):
        """
        initialize state
        """
        self.player_id: int = int(messages[0])
        self.turn: int = -1
        # get some other necessary initial input
        mapInfo = messages[1].split(" ")
        self.map_width: int = int(mapInfo[0])
        self.map_height: int = int(mapInfo[1])
        self.map: GameMap = GameMap(self.map_width, self.map_height)
        self.players: List[Player] = [Player(0), Player(1)]

        self.resource_scores_matrix: List[List[float]] = None
        self.resource_rates_matrix: List[List[float]] = None
        self.maxpool_scores_matrix: List[List[float]] = None
        self.city_tile_matrix: List[List[float]] = None
        self.empty_tile_matrix: List[List[float]] = None

    def _end_turn(self):
        print("D_FINISH")

    def _reset_player_states(self):
        self.players[0].units = []
        self.players[0].cities = {}
        self.players[0].city_tile_count = 0
        self.players[1].units = []
        self.players[1].cities = {}
        self.players[1].city_tile_count = 0

        self.player = self.players[self.player_id]
        self.opponent = self.players[1 - self.player_id]

    def _update(self, messages):
        """
        update state
        """
        self.map = GameMap(self.map_width, self.map_height)
        self.turn += 1
        self._reset_player_states()

        for update in messages:
            if update == "D_DONE":
                break
            strs = update.split(" ")
            input_identifier = strs[0]
            if input_identifier == INPUT_CONSTANTS.RESEARCH_POINTS:
                team = int(strs[1])
                self.players[team].research_points = int(strs[2])

            elif input_identifier == INPUT_CONSTANTS.RESOURCES:
                r_type = strs[1]
                x = int(strs[2])
                y = int(strs[3])
                amt = int(float(strs[4]))
                self.map._setResource(r_type, x, y, amt)

            elif input_identifier == INPUT_CONSTANTS.UNITS:
                unittype = int(strs[1])
                team = int(strs[2])
                unitid = strs[3]
                x = int(strs[4])
                y = int(strs[5])
                cooldown = float(strs[6])
                wood = int(strs[7])
                coal = int(strs[8])
                uranium = int(strs[9])
                self.players[team].units.append(
                    Unit(team, unittype, unitid, x, y, cooldown, wood, coal, uranium))

            elif input_identifier == INPUT_CONSTANTS.CITY:
                team = int(strs[1])
                cityid = strs[2]
                fuel = float(strs[3])
                lightupkeep = float(strs[4])
                self.players[team].cities[cityid] = City(
                    team, cityid, fuel, lightupkeep)

            elif input_identifier == INPUT_CONSTANTS.CITY_TILES:
                team = int(strs[1])
                cityid = strs[2]
                x = int(strs[3])
                y = int(strs[4])
                cooldown = float(strs[5])
                city = self.players[team].cities[cityid]
                citytile = city._add_city_tile(x, y, cooldown)
                self.map.get_cell(x, y).citytile = citytile
                self.players[team].city_tile_count += 1

            elif input_identifier == INPUT_CONSTANTS.ROADS:
                x = int(strs[1])
                y = int(strs[2])
                road = float(strs[3])
                self.map.get_cell(x, y).road = road

        # update matrices
        self.calculate_matrix()
        self.calculate_resource_scores_and_rates_matrix()
        self.calculate_resource_maxpool_matrix()

        self.player.make_index_units_by_id()
        self.opponent.make_index_units_by_id()

    def calculate_matrix(self) -> None:
        def init_zero_matrix():
            return [[0 for _ in range(self.map_height) for _ in range(self.map_width)]]

        self.empty_tile_matrix = init_zero_matrix()

        self.wood_amount_matrix = init_zero_matrix()
        self.coal_amount_matrix = init_zero_matrix()
        self.uranium_amount_matrix = init_zero_matrix()

        self.player_city_tile_matrix = init_zero_matrix()
        self.opponent_city_tile_matrix = init_zero_matrix()

        self.player_units_matrix = init_zero_matrix()
        self.opponent_units_matrix = init_zero_matrix()

        self.empty_tile_matrix = init_zero_matrix()

        for y in range(self.map_width):
            for x in range(self.map_height):
                cell = self.map.get_cell(x, y)

                if cell.has_resource():
                    if cell.resource.type == RESOURCE_TYPES.WOOD:
                        self.wood_amount_matrix[y][x] += cell.resource.amount
                    if cell.resource.type == RESOURCE_TYPES.COAL:
                        self.coal_amount_matrix[y][x] += cell.resource.amount
                    if cell.resource.type == RESOURCE_TYPES.URANIUM:
                        self.uranium_amount_matrix[y][x] += cell.resource.amount

                elif cell.citytile:
                    if cell.citytile.team == self.player_id:
                        self.player_city_tile_matrix[y][x] += 1
                    else:   # city tile belongs to opponent
                        self.opponent_city_tile_matrix[y][x] += 1

                elif cell.unit:
                    if cell.unit.team == self.player_id:
                        self.player_units_matrix[y][x] += 1
                    else:   # unit belongs to opponent
                        self.opponent_units_matrix[y][x] += 1

                else:
                    self.empty_tile_matrix[y][x] += 1

        self.convert_into_sets()
    
    def convert_into_sets(self):
        # or should we use dict?
        self.empty_tile_xy_set = set()
        self.wood_amount_xy_set = set()
        self.coal_amount_xy_set = set()
        self.uranium_amount_xy_set = set()
        self.player_city_tile_xy_set = set()
        self.opponent_city_tile_xy_set = set()
        self.player_units_xy_set = set()
        self.opponent_units_xy_set = set()
        self.empty_tile_xy_set = set()

        for set_object, matrix in [
            [self.empty_tile_xy_set,            self.empty_tile_matrix],
            [self.wood_amount_xy_set,           self.wood_amount_matrix],
            [self.coal_amount_xy_set,           self.coal_amount_matrix],
            [self.uranium_amount_xy_set,        self.uranium_amount_matrix],
            [self.player_city_tile_xy_set,      self.player_city_tile_matrix],
            [self.opponent_city_tile_xy_set,    self.opponent_city_tile_matrix],
            [self.player_units_xy_set,          self.player_units_matrix],
            [self.opponent_units_xy_set,        self.opponent_units_matrix],
                [self.empty_tile_xy_set,            self.empty_tile_matrix]]:

            for y in range(self.map.width):
                for x in range(self.map.height):
                    if matrix[y][x] > 0:
                        set_object.add((x, y))

        self.map.set_occupied_xy = (self.player_units_xy_set | self.opponent_units_xy_set | self.player_city_tile_xy_set) \
            - self.player_city_tile_xy_set

    def calculate_resource_scores_and_rates_matrix(self) -> None:
        width, height = self.map_width, self.map_height
        player = self.player
        resource_scores_matrix = [
            [0 for _ in range(width) for _ in range(height)]]
        resource_rates_matrix = [
            [0 for _ in range(width)] for _ in range(height)]

        for y in range(height):
            for x in range(width):
                resource_scores_cell = 0
                resource_rates_cell = 0
                for dx, dy in [(0, 0), (0, -1), (0, 1), (1, 0), (-1, 0)]:
                    xx, yy = x+dx, y+dy
                    if 0 <= xx < width and 0 <= yy < height:
                        cell = self.map.get_cell(xx, yy)
                        if not cell.has_resource():
                            continue
                        if not player.researched_coal() and cell.resource.type == RESOURCE_TYPES.COAL:
                            continue
                        if not player.researched_uranium() and cell.resource.type == RESOURCE_TYPES.URANIUM:
                            continue

                        fuel = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_TO_FUEL_RATE"][str(
                            cell.resource.type).upper()]
                        mining_rate = GAME_CONSTANTS["PARAMETERS"]["WORKER_COLLECTION_RATE"][str(
                            cell.resource.type).upper()]

                        resource_scores_cell += fuel * cell.resource.amount
                        resource_rates_cell += fuel * mining_rate

                resource_scores_matrix[y][x] = resource_scores_cell
                resource_rates_matrix[y][x] = resource_rates_cell

        self.resource_scores_matrix = resource_scores_matrix
        self.resource_rates_matrix = resource_rates_matrix

    def calculate_resource_maxpool_matrix(self) -> None:
        width, height = self.map_width, self.map_height
        maxpool_scores_matrix = [
            [0 for _ in range(width) for _ in range(height)]]

        for y in range(height):
            for x in range(width):
                for dx, dy in [(1, 0), (0, 1), (0, 0), (-1, 0), (0, -1)]:
                    xx, yy = x+dx, y+dy
                    if not (0 <= xx < width and 0 <= yy < height):
                        continue
                    if self.resource_scores_matrix[yy][xx] + dx * 0.2 + dy * 0.1 > self.resource_scores_matrix[x][y]:
                        break
                    else:
                        maxpool_scores_matrix[y][x] = self.resource_scores_matrix[y][x]

        self.maxpool_scores_matrix = maxpool_scores_matrix

    def get_city_tile_matrix(self) -> List[List[int]]:
        width, height = self.map_width, self.map_height
        player = self.player
        city_tile_matrix = [
            [0 for _ in range(width) for _ in range(height)]]

        for _, city in player.cities.items():
            for city_tile in city.citytiles:
                city_tile_matrix[city_tile.pos.y][city_tile.pos.x] += 1

        return city_tile_matrix

    def get_empty_tile_matrix(self) -> List[List[int]]:
        width, height = self.map_width, self.map_height
        empty_tile_matrix = [
            [0 for _ in range(width) for _ in range(height)]]

        for y in range(height):
            for x in range(width):
                cell = self.map.get_cell(x, y)
                if cell.has_resource():
                    continue
                if cell.citytile:
                    continue
                empty_tile_matrix[y][x] = 1

        return empty_tile_matrix

    def get_nearest_empty_tile(self, current_position: Position) -> Tuple[Position, int]:
        width, height = self.map_width, self.map_height

        nearest_distance = width + height
        nearest_position: Position = None

        for y in range(height):
            for x in range(width):
                if self.empty_tile_matrix[y][x] == 0:  # not empty
                    position = Position(x, y)
                    distance = position - current_position
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_position = position

        return nearest_position, nearest_distance

    def calculate_occupied_and_free_zones(self) -> None:
        player, opponent = self.player, self.opponent

        set_occupied_xy = set()
        set_player_city_tiles_xy = set()

        for city in player.cities.values():
            for city_tile in city.citytiles:
                xy = (city_tile.pos.x, city_tile.pos.y)
                set_player_city_tiles_xy.add(xy)

        for unit in player.units:
            xy = (unit.pos.x, unit.pos.y)
            if xy in set_player_city_tiles_xy:
                continue
            set_occupied_xy.add(xy)

        for city in opponent.cities.values():
            for city_tile in city.citytiles:
                xy = (city_tile.pos.x, city_tile.pos.y)
                set_occupied_xy.add(xy)

        for unit in opponent.units:
            xy = (unit.pos.x, unit.pos.y)
            set_occupied_xy.add(xy)

        self.map.set_occupied_xy = set_occupied_xy
        self.map.set_player_city_tiles_xy = set_player_city_tiles_xy
