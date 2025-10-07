# !/usr/env python

from random import random, seed, randint
import math

import numpy as np
from abviralsimulator.decorators import Actor, dying_probability, transmission_probability, movement, \
    vaccination_priority
from abviralsimulator.decorators import home_location, Scenario, at_home, at_home_infected, at_home_diagnosed, \
    initially_infected, testing_probability
from abviralsimulator.decorators import Virus, duration, showing_symphoms_after, asymptomatic
from abviralsimulator.decorators import Scenario
from abviralsimulator.decorators import daily_vaccines, available_tests
from abviralsimulator.store import DataStore
from abviralsimulator.helpers import Common
from abviralsimulator.enums import ViralState
from abviralsimulator.utility import Plotter
import matplotlib.pyplot as plt
from abviralsimulator import Simulator

box_size = 3000
VELOCITY = 0.4


@Virus(name="covid")
class Covid:
    infectious_radius = 1  # meters

    @duration
    def get_duration(self, time_step):
        base_duration = 14
        duration_in_days = base_duration + randint(0, 7)

        return math.ceil(duration_in_days * 24 * 60 / time_step)

    @showing_symphoms_after
    def get_duration_after_symptoms_show(self, time_step):
        base_duration = 4
        duration_in_days = base_duration + randint(0, 2)
        return math.ceil(duration_in_days * 24 * 60 / time_step)

    @asymptomatic
    def get_asymptomatic_rate(self):
        return 0.2


@Actor(name="person")
class Person:

    def __init__(self, location, is_good_citizen, age_group, virus_type):
        self.location = location
        self.is_good_citizen = is_good_citizen
        self.age_group = age_group
        self.virus_type = virus_type

    @home_location
    def get_home_location(self):
        return self.location

    @dying_probability(Covid)
    def get_covid_dying_probability(self):
        return 0.02

    @transmission_probability(Covid, initial_probability=0, max_probability=0.1)
    def transmission_prbability_covid(self, transmission_probability, time_step):
        return transmission_probability + 0.01 * time_step / (24 * 60)

    def random_walk(self, home_location, time_step):

        gyration_radius = 3800  # in meters
        velocity = VELOCITY  # m/s

        current_position = home_location
        magnitude = velocity * 60 * time_step

        while True:
            x, y = current_position
            p = random() * 2 * np.pi
            r = random() * magnitude  # time_step is the number of minutes per time step and velocity is in m/s
            x += np.cos(p) * r
            y += np.sin(p) * r

            vec = x - home_location[0], y - home_location[1]
            mag = vec[0] ** 2 + vec[1] ** 2

            if mag < gyration_radius ** 2:
                current_position = (x, y)

            yield current_position

    def random_walk_unbounded(self, home_location, time_step):

        velocity = VELOCITY  # m/s

        current_position = home_location
        magnitude = velocity * 60 * time_step

        while True:
            x, y = current_position
            p = random() * 2 * np.pi
            r = random() * magnitude  # time_step is the number of minutes per time step and velocity is in m/s
            x += np.cos(p) * r
            y += np.sin(p) * r

            current_position = (x, y)

            yield current_position

    @movement
    def random_walk_bound_box(self, home_location, time_step):
        velocity = VELOCITY  # m/s

        current_position = home_location
        magnitude = velocity * 60 * time_step
        while True:
            x, y = current_position
            p = random() * 2 * np.pi
            r = random() * magnitude  # time_step is the number of minutes per time step and velocity is in m/s
            x += np.cos(p) * r
            y += np.sin(p) * r

            x = Common.clamp(x, 0, box_size)
            y = Common.clamp(y, 0, box_size)

            current_position = (x, y)

            yield current_position

    def random_walk_reflective_box(self, home_location, time_step):
        velocity = VELOCITY  # m/s

        current_position = home_location
        magnitude = velocity * 60 * time_step
        while True:
            x, y = current_position
            p = random() * 2 * np.pi
            r = random() * magnitude  # time_step is the number of minutes per time step and velocity is in m/s
            delta_x = np.cos(p) * r
            delta_y = np.sin(p) * r

            if x + delta_x > box_size or x + delta_x < 0:
                delta_x = -delta_x

            if y + delta_y > box_size or y + delta_y < 0:
                delta_y = -delta_y

            x += delta_x
            y += delta_y
            current_position = (x, y)

            yield current_position

    def random_walk_true_reflective_box(self, home_location, time_step):
        velocity = VELOCITY  # m/s

        current_position = home_location
        magnitude = velocity * 60 * time_step
        while True:
            x, y = current_position
            p = random() * 2 * np.pi
            r = random() * magnitude  # time_step is the number of minutes per time step and velocity is in m/s
            delta_x = np.cos(p) * r
            delta_y = np.sin(p) * r


            reflected_x = 0
            # Reflect on the right side
            if x + delta_x > box_size:
                reflected_x = box_size - (x + delta_x)
                delta_x = box_size - x
            # Reflect on the left side
            elif x + delta_x < 0:
                reflected_x = -(x + delta_x)
                delta_x = 0 - x


            reflected_y = 0
            # Reflect on the top
            if y + delta_y > box_size:
                reflected_y = box_size - (y + delta_y)
                delta_y = box_size - y
            # Reflect on the bottom
            elif y + delta_y < 0:
                reflected_y = -(y + delta_y)
                delta_y = 0 - y


            x += (delta_x + reflected_x)
            y += (delta_y + reflected_y)
            current_position = (x, y)

            yield current_position


    @at_home
    def get_staying_at_home_duration(self):
        return 8

    @at_home_infected(Covid)
    def get_staying_at_home_covid_duration(self):
        if self.is_good_citizen:
            return 22
        else:
            return 10

    @at_home_diagnosed(Covid)
    def get_staying_at_home_covid_diagnosed_duration(self):
        if self.is_good_citizen:
            return 22
        else:
            return 10

    @initially_infected
    def get_infection(self):
        return self.virus_type

    @testing_probability(Covid)
    def get_covid_testing_probability(self):
        return 0.7

    @vaccination_priority
    def get_priority(self):
        return 1


class PublicHealthcare:

    def __init__(self, available_tests_count):
        self.available_tests_count = available_tests_count
        self.vaccine_daily_count = {}

    def add_daily_count(self, vaccine_type, daily_count):
        self.vaccine_daily_count[vaccine_type] = daily_count

    @daily_vaccines
    def get_daily_count(self, vaccine_type):
        return self.vaccine_daily_count[vaccine_type]

    @available_tests
    def get_available_tests(self):
        return self.available_tests_count


@Scenario(name="small", time_step=15, iterations=90 * 24 * 60 // 15)
class SmallScenario:

    def setup_healthcare(self):
        healthcare = PublicHealthcare(12000)

        return healthcare

    def setup(self):

        agents = []
        seed(2021)
        infected_count = 0
        for _ in range(12000):
            virus_type = Covid if random() < 0.01 else None
            if virus_type is not None:
                infected_count += 1
            good_citizen = True
            agent = Person(location=(box_size * random(), box_size * random()), is_good_citizen=good_citizen, age_group=1,
                           virus_type=virus_type)
            agents.append(agent)
        print(infected_count)
        return agents


simulator = Simulator()
simulator.run()

statistics = DataStore()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

Plotter.draw_bar_graph(statistics.contact_count.daily_contact_count, plot=ax1, title="Contacts per day")
Plotter.draw_bar_graph(statistics.contact_count.daily_unique_contact_count, plot=ax2, title="Unique contacts per day")
Plotter.draw_bar_graph(statistics.traveled_distance.traveled_distance, plot=ax3, title="Covered ground per day")
Plotter.draw_active_infected(statistics.state_changes.state_changes, plot=ax4, plot_days=True, title="Infected per day")

plt.show()
