from enum import IntEnum


class ViralState(IntEnum):
    UNAFFECTED = 1
    INFECTED = 2
    DIAGNOSED = 3
    RECOVERED = 4
    IMMUNE = 5
    DECEASED = 6
