## **Make submission files**

Run `./submission.sh` to generate the following files

- The zipfile `submission.tar.gz` which can be submitted to Kaggle [directly](https://www.kaggle.com/c/lux-ai-2021/submissions).
- The notebook submission `notebook_generated.ipynb`. You can upload this notebook into a [fork of my notebook](https://www.kaggle.com/kernels/fork-version/73552476) and submit.


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