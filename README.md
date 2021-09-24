# **LUX Deep Reinforcement Learning**
Kaggle Lux AI competition

## **Setup**

Run `./setup.sh` to install the required components.

You will need Node version 12 or above, and conda, to install the script.


## **Make submission files**

Run `./submission.sh` to generate the following files

- The zipfile `submission.tar.gz` which can be submitted to Kaggle [directly](https://www.kaggle.com/c/lux-ai-2021/submissions).
- The notebook submission `notebook_generated.ipynb`. You can upload this notebook into a [fork of my notebook](https://www.kaggle.com/kernels/fork-version/73552476) and submit.


## **API**
<details>
<summary> <b>Game API details</b> </summary>

### game_state

The `game_state` object is provided to you and contains the complete information about the current state of the game at the current turn `game_state.turn`. Each agent/player in the game has an id, with your bot's id equal to `game_state.id` and the other team's id being `(game_state.id + 1) % 2`.

Additionally in the game state, are the following nested objects, `map` of type [GameMap](#GameMap), and `players` which is a list with two [Player](#Player) objects indexed by the player's team id. The kits will show how to retrieve those objects. The rest of this section details the properties and methods of each type of object used in the kits.

### <u>GameMap</u>

The map is organized such that the top left corner of the map is at `(0, 0)` and the bottom right is at `(width, height)`. The map is always square.

Properties:

- `height: int` - the height of the map (along the y direction)
- `width: int` - the width of the map (along the x direction)
- `map: List[List[Cell]]` - A 2D array of [Cell](#Cell) objects, defining the current state of the map. `map[y][x]` represents the cell at coordinates (x, y) with `map[0][0]` being the top left Cell.

Methods:

- `get_cell_by_pos(pos: Position) -> Cell` - returns the [Cell](#Cell) at the given pos
- `get_cell(x: int, y: int) -> Cell` - returns the [Cell](#Cell) at the given x, y coordinates

### <u>Position</u>

Properties:

- `x: int` - the x coordinate of the [Position](#Position)
- `y: int` - the y coordinate of the [Position](#Position)

Methods:

- `is_adjacent(pos: Position) -> bool` - returns true if this [Position](#Position) is adjacent to `pos`. False otherwise

- `equals(pos: Position) -> bool` - returns true if this [Position](#Position) is equal to the other `pos` object by checking x, y coordinates. False otherwise

- `translate(direction: DIRECTIONS, units: int) -> Position` - returns the [Position](#Position) equal to going in a `direction` `units` number of times from this [Position](#Position)

- `distance_to(pos: Position) -> float` - returns the [Manhattan (rectilinear) distance](https://en.wikipedia.org/wiki/Taxicab_geometry) from this [Position](#Position) to `pos`

- `direction_to(target_pos: Position) -> DIRECTIONS` - returns the direction that would move you closest to `target_pos` from this [Position](#Position) if you took a single step. In particular, will return `DIRECTIONS.CENTER` if this [Position](#Position) is equal to the `target_pos`. Note that this does not check for potential collisions with other units but serves as a basic pathfinding method

### <u>Cell</u>

Properties:

- `pos: Position`
- `resource: Resource` - contains details of a Resource at this [Cell](#Cell). This may be equal to `None` or `null` equivalents in other languages. You should always use the function `has_resource` to check if this [Cell](#Cell) has a Resource or not
- `road: float` - the amount of Cooldown subtracted from a [Unit's](#Unit) Cooldown whenever they perform an action on this tile. If there are roads, the more developed the road, the higher this Cooldown rate value is. Note that a [Unit](#Unit) will always gain a base Cooldown amount whenever any action is performed.
- `citytile: CityTile` - the citytile that is on this [Cell](#Cell). Equal to `none` or `null` equivalents in other languages if there is no [CityTile](#CityTile) here.

Methods:

- `has_resource() -> bool` - returns true if this [Cell](#Cell) has a non-depleted Resource, false otherwise

### <u>City</u>

Properties:

- `cityid: str` - the id of this [City](#City). Each [City](#City) id in the game is unique and will never be reused by new cities
- `team: int` - the id of the team this [City](#City) belongs to.
- `fuel: float` - the fuel stored in this [City](#City). This fuel is consumed by all CityTiles in this [City](#City) during each turn of night.
- `citytiles: list[CityTile]` - a list of [CityTile](#CityTile) objects that form this one [City](#City) collectively. A [City](#City) is defined as all CityTiles that are connected via adjacent CityTiles.

Methods:

- `get_light_upkeep() -> float` - returns the light upkeep per turn of the [City](#City). Fuel in the [City](#City) is subtracted by the light upkeep each turn of night.

### <u>CityTile</u>

Properties:

- `cityid: str` - the id of the [City](#City) this [CityTile](#CityTile) is a part of. Each [City](#City) id in the game is unique and will never be reused by new cities
- `team: int` - the id of the team this [CityTile](#CityTile) belongs to.
- `pos: Position` - the [Position](#Position) of this [City](#City) on the map
- `cooldown: float` - the current Cooldown of this [City](#City).

Methods:

- `can_act() -> bool` - whether this [City](#City) can perform an action this turn, which is when the Cooldown is less than 1

- `research() -> str` - returns the research action

- `build_worker() -> str` - returns the build worker action. When applied and requirements are met, a worker will be built at the [City](#City).

- `build_cart() -> str` - returns the build cart action. When applied and requirements are met, a cart will be built at the [City](#City).

### <u>Unit</u>

Properties:

- `pos: Position` - the [Position](#Position) of this [Unit](#Unit) on the map
- `team: int` - the id of the team this [Unit](#Unit) belongs to.
- `id: str` - the id of this [Unit](#Unit). This is unique and cannot be repeated by any other [Unit](#Unit) or [City](#City)
- `cooldown: float` - the current Cooldown of this [Unit](#Unit). Note that when this is less than 1, the [Unit](#Unit) can perform an action
- `cargo.wood: int` - the amount of wood held by this [Unit](#Unit)
- `cargo.coal: int` - the amount of coal held by this [Unit](#Unit)
- `cargo.uranium: int` - the amount of uranium held by this [Unit](#Unit)

Methods:

- `get_cargo_space_left(): int` - returns the amount of space left in the cargo of this [Unit](#Unit). Note that any Resource takes up the same space, e.g. 70 wood takes up as much space as 70 uranium, but 70 uranium would produce much more fuel than wood when deposited at a [City](#City)
- `can_build(game_map: GameMap): bool` - returns true if the [Unit](#Unit) can build a [City](#City) on the tile it is on now. False otherwise. Checks that the tile does not have a Resource over it still and the [Unit](#Unit) has a Cooldown of less than 1
- `can_act(): bool`  - returns true if the [Unit](#Unit) can perform an action. False otherwise. Essentially checks whether the Cooldown of the [Unit](#Unit) is less than 1
- `move(dir): str` - returns the move action. When applied, [Unit](#Unit) will move in the specified direction by one [Unit](#Unit), provided there are no other units in the way or opposition cities. ([Units](#Unit) can stack on top of each other however when over a friendly [City](#City))
- `transfer(dest_id, resourceType, amount): str` - returns the transfer action. Will transfer from this [Unit](#Unit) the selected Resource type by the desired amount to the [Unit](#Unit) with id `dest_id` given that both units are adjacent at the start of the turn. (This means that a destination [Unit](#Unit) can receive a transfer of resources by another [Unit](#Unit) but also move away from that [Unit](#Unit))
- `build_city(): str` - returns the build [City](#City) action. When applied, [Unit](#Unit) will try to build a [City](#City) right under itself provided it is an empty tile with no [City](#City) or resources and the worker is carrying 100 units of resources. All resources are consumed if the city is succesfully built.
- `pillage(): str` - returns the pillage action. When applied, [Unit](#Unit) will pillage the tile it is currently on top of and remove 0.5 of the road level.

### <u>Player</u>

This contains information on a particular player of a particular team.

Properties:

- `team: int` - the team id of this player

- `research_points: int` - the current total number of research points the player's team has
- `units: list[Unit]` - a list of every [Unit](#Unit) owned by this player's team.
- `cities: Dict[str, City]` - a dictionary / map mapping [City](#City) id to each separate [City](#City) owned by this player's team. To get the individual CityTiles, you will need to access the `citytiles` property of the [City](#City).

Methods:

- `researched_coal() - bool` - whether or not this player's team has researched coal and can mine coal.
- `researched_uranium() - bool` - whether or not this player's team has researched uranium and can mine uranium.

### <u>Annotate</u>

The annotation object lets you create annotation commands that show up on the visualizer when debug mode is turned on. Note that these commands are stripped by competition servers but are available to see when running matches locally.

Methods

- `circle(x: int, y: int) -> str` - returns the draw circle annotation action. Will draw a unit sized circle on the visualizer at the current turn centered at the [Cell](#Cell) at the given x, y coordinates

- `x(x: int, y: int) -> str` - returns the draw X annotation action. Will draw a unit sized X on the visualizer at the current turn centered at the [Cell](#Cell) at the given x, y coordinates

- `line(x1: int, y1: int, x2: int, y2: int) -> str` - returns the draw line annotation action. Will draw a line from the center of the [Cell](#Cell) at (x1, y1) to the center of the [Cell](#Cell) at (x2, y2)

- `text(x: int, y: int, message: str, fontsize: int = 16) -> str:` - returns the draw text annotation action. Will write text on top of the tile at (x, y) with the particular message and fontsize

- `sidetext(message: str) -> str:` - returns the draw side text annotation action. Will write text that is displayed on that turn on the side of the visualizer

Note that all of these will be colored according to the team that created the annotation (blue or orange)

### <u>GameConstants</u>

This will contain constants on all game parameters like the max turns, the light upkeep of CityTiles etc.

If there are any crucial changes to the starter kits, typically only this object will change.

</details>

## **Running a Game**

Run a game with `notebook_test.ipynb`


## **Analysing Game State and Missions**

Run the game logic on a specific turn with `notebook_debug.ipynb` to understand why the agent has made a certain decision.

---

When the run is made with notebook_test.ipynb, the game state, and missions are saved as Python pickle files. You may load the Python pickle file and run the game logic to produce the actions and updated missions.

As the game logic is run, it prints the new mission planned, the actions made include the annotation.

You can modify the code and see how the agent reacts different for the same game state and missions. This allows you iteratively improve the agent more quicking.

## **Game Details**
#### The objective

- At the end of 360 turns (the player first without units loses)
  - The player with the most city tiles wins.
  - The only tie breaker is the number of units.


#### Rules that affect implementation

- When a unit enters your city, the resources are automatically transformed into city fuel. Transferring is meant for transferring between your units.
- When a unit is beside a resource, it collects the resource passively. No action is required.
- Fuel is automatically shared between connected cities.
- Workers do not and cannot carry fuel.
  - Workers cannot move out of a city at night if there are no adjacent resources. It will die.
- It is not possible to build a house at night. (The game resolution order does not allow that. Between resource collection and unit action, resources are consumed for the night. It is not possible to have full resources that is needed to build a house.)
- A cart can mine resources like a worker. A cart cannot pillage or build a city.
- You cannot build a city and make units in the same turn. (See game resolution order)



### Rules that affect strategy

- When a city exist, it does not cost anything to produce a worker, cart or research.
- Only adjacent cities share fuel. City tiles that are not connected does not share fuel.
- When a city tile has enough fuels to last till turn 360, it will definitely last till turn 360. It might die only if you build an adjacent house.
- A city tile can immediately take an action once it is built.


### Overall Strategy

- We treat cities and units as disposable.
- Do not bother supporting the cities over the first few nights. Building a house takes 100 resources units, whereas sustaining over the night takes 300 fuels.
- At the mid game, start securing cities to last for the entire game. Do not create cities beside one another, for now, for easier calculation.



### Procedures

- Every time the unit is full and it is going to be day for the next few days, build a city.
- Once a city is built or cooldown is ready (done)
  - If researched uranium, build unit
  - If haven't researched uranium, haven't exceed the limit  build a unit.
  - Otherwise, do research.
- Once a unit is built
  - It goes to to the best cluster.
- Once a unit is full and it is going to be day for the next few days.
  - Build a city.



### Future roadmap

- (Necessary for competitive submission) When resources are depleting - secure city tiles by providing fuel to last through the remaining nights.
- Consider the use of carts
- Collision avoidance
- Offensive strategies (start with interfering opponent operations)
- Hyperparamter tuning (probably the only ML part, and probably only using Bayesian optimization)



### Roadmap

- Nomad strategy (the "procedures")
- Securing city tiles
- Collision avoidance



#### Functions

- How are clusters defined? What is meant by a good cluster?
  - Cluster scores are defined from a point of view.
  - A group of four resources is considered a cluster.


## **References**
- https://github.com/tonghuikang/lux-ai-2021
- https://github.com/glmcdona/LuxPythonEnvGym/