import math
from random import random
from abviralsimulator.enums import MovementState


class Movement:

    def __init__(self, movement_handler, transition_rules, time_step):
        self.movement_state = MovementState.STAYING_AT_HOME
        self.state_change_counter = 0
        self.movement_handler = movement_handler
        self.movement_sequence = movement_handler()
        self.transition_rules = transition_rules
        self.time_step = time_step
        self.__update_state_counter(None, None)

    def __update_state_counter(self, viral_state, virus_type):
        if viral_state in self.transition_rules:
            duration = self.transition_rules[viral_state][virus_type]()
        else:
            duration = self.transition_rules[None][None]()
        if self.movement_state == MovementState.STAYING_AT_HOME:
            duration = 24 - duration
        # self.state_change_counter = math.ceil(-math.log(1 - random()) * duration * 60/self.time_step)
        self.state_change_counter = math.ceil(duration * 60 / self.time_step)

    def change(self):
        if self.movement_state == MovementState.STAYING_AT_HOME:
            self.movement_state = MovementState.MOVING
            self.movement_sequence = self.movement_handler()
        else:
            self.movement_state = MovementState.STAYING_AT_HOME

    def get_next_location(self):
        return next(self.movement_sequence)

    def is_moving(self):
        return self.movement_state == MovementState.MOVING

    def update(self, viral_state, virus_type):

        if self.state_change_counter > 0:
            self.state_change_counter -= 1
        else:
            self.__update_state_counter(viral_state, virus_type)
            self.change()
