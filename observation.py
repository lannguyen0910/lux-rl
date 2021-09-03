import numpy as np
from numpy.lib.shape_base import split


class Observation():
    def __init__(self, observation):
        self.observation = observation

        self.wood_map = np.zeros((32, 32))
        self.coal_map = np.zeros((32, 32))
        self.uran_map = np.zeros((32, 32))

        self.worker_cooldown = np.zeros((2, 32, 32))
        self.worker_capacity = np.zeros((2, 32, 32))

        self.cart_cooldown = np.zeros((2, 32, 32))
        self.cart_capacity = np.zeros((2, 32, 32))

        self.city_tiles_cooldown = np.zeros((2, 32, 32))
        self.city_tiles_fuel = np.zeros((2, 32, 32))

        self.step = self.observation[0]["observation"]["step"]

        pad_size_width = (
            32 - self.observation[0]["observation"]["width"]) // 2
        pad_size_length = (
            32 - self.observation[0]["observation"]["height"]) // 2

        self.worker_pos_dict = {}
        self.ct_pos_dict = {}

        for player in range(1):
            ups = observation[player]["observation"]["updates"]
            cities = {}
            for row in ups:
                splits = row.split(" ")
                if splits[0] == "r":
                    if splits[1] == "wood":
                        self.wood_map[(
                            int(splits[2]) + pad_size_width,
                            int(splits[3]) + pad_size_length)] = int(float(splits[4]))

                    elif splits[1] == "uranium":
                        self.uran_map[(
                            int(splits[2]) + pad_size_width,
                            int(splits[3]) + pad_size_length)] = int(float(splits[4]))

                    elif splits[1] == "coal":
                        self.coal_map[(
                            int(splits[2]) + pad_size_width,
                            int(splits[3]) + pad_size_length)] = int(float(splits[4]))

                elif splits[0] == "c":
                    cities[splits[2]] = int(splits[3])

                elif splits[0] == "u":
                    self.worker_capacity[(
                        int(splits[2]),
                        int(splits[4]) + pad_size_width,
                        int(splits[5]) + pad_size_length
                    )] = int(splits[7]) + int(splits[8]) + int(splits[9])
                    self.worker_cooldown[(
                        int(splits[2]),
                        int(splits[4]) + pad_size_width,
                        int(splits[5]) + pad_size_length
                    )] = int(splits[6])
                    self.worker_pos_dict[(
                        int(splits[4]) + pad_size_width,
                        int(splits[5]) + pad_size_length)] = splits[3]

                elif splits[0] == "ct":
                    city_fuel = cities.get(splits[2])
                    self.city_tiles_cooldown[(
                        int(splits[1]),
                        int(splits[3]) + pad_size_width,
                        int(splits[4]) + pad_size_length)] = int(splits[5])
                    self.city_tiles_fuel[(
                        int(splits[1]),
                        int(splits[3]) + pad_size_width,
                        int(splits[4]) + pad_size_length)] = int(city_fuel)
                    self.ct_pos_dict[(
                        int(splits[3]) + pad_size_width,
                        int(splits[4]) + pad_size_length)] = splits[2]

        self.wood_map = np.expand_dims(self.wood_map, axis=0)
        self.uran_map = np.expand_dims(self.uran_map, axis=0)
        self.coal_map = np.expand_dims(self.coal_map, axis=0)

        self.state = np.concatenate((
            self.wood_map / 1000, self.uran_map / 1000, self.coal_map / 1000,
            self.worker_cooldown / 2, self.worker_capacity / 100,
            self.city_tiles_fuel / 1000, self.city_tiles_cooldown / 10), axis=0)


def log_to_action(entity_action_prob, is_worker=True):
    entity_action_dim = {
        0: "n",
        1: "s",
        2: "w",
        3: "e",
        4: "stay",
        5: "bcity",
        6: "bw",
        7: "r",
        8: "None"
    }

    if is_worker:
        ordered_actions = [(entity_action_dim[i], entity_action_prob[i])
                           for i in range(6)]
    else:
        ordered_actions = [(entity_action_dim[i], entity_action_prob[i])
                           for i in range(6, 9)]

        ordered_actions = sorted(
            ordered_actions, key=lambda x: x[1], reverse=True)

    return ordered_actions
