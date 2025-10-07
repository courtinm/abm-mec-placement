import math


class Algebra:

    @staticmethod
    def euclid_distance(x, y):
        return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
