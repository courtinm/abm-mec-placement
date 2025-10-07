import operator
from functools import reduce
from scipy import spatial


class LocationQuery:

    data = {}
    kd_trees = {}
    probabilities = {}

    @classmethod
    def setup(cls, viruses):
        for virus_type in viruses.values():
            cls.data[virus_type] = []

    @classmethod
    def update(cls):
        for virus_type, people in cls.data.items():
            if len(people) == 0:
                cls.kd_trees[virus_type] = None
                cls.probabilities[virus_type] = None
                continue

            locations, probabilities = zip(*people)
            cls.kd_trees[virus_type] = spatial.KDTree(locations)
            cls.probabilities[virus_type] = probabilities

        for virus_type in cls.data.keys():
            cls.data[virus_type] = []

    @classmethod
    def query(cls, location):
        infection_probabilties = []
        for virus_type, tree in cls.kd_trees.items():
            if tree is None:
                continue
            indexes = tree.query_ball_point(location, virus_type.infectious_radius)

            probabilities = [cls.probabilities[virus_type][index] for index in indexes]
            probability = 1 - reduce(operator.mul, [1 - probability for probability in probabilities], 1)

            infection_probabilties.append((probability, virus_type))

        return infection_probabilties

    @classmethod
    def add(cls, virus_type, location, transmission_probability):
        cls.data[virus_type].append((location, transmission_probability))
