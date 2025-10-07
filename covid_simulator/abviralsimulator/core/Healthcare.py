from random import random

from abviralsimulator.core import Handler
from abviralsimulator.enums import HealthcareHandlers
from abviralsimulator.structure import Singleton


class Healthcare(metaclass=Singleton):

    def __init__(self,healthcare=-1):
        if healthcare != -1:
            print("called")
            available_tests_handler = Handler.get_handler(healthcare, HealthcareHandlers.AVAILABLE_TESTS)

            self.total_test_count = available_tests_handler()
        else:
            print("it is -1")
            self.total_test_count = 0

    def has_available_tests(self, virus_type):
        return self.total_test_count != 0

    def is_getting_tested(self, testing_probability):
        if self.total_test_count == 0:
            return False

        if random() < testing_probability:
            return True

    def get_tested(self):
        self.total_test_count -= 1
