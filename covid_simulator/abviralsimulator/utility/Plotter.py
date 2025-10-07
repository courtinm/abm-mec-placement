from abviralsimulator.enums import ViralState

import numpy as np
import matplotlib.pyplot as plt


class Plotter:

    @staticmethod
    def to_days(time_step, infected):
        iteration_in_day = 24*60//time_step
        data = []
        while len(infected) != 0:
            data = data + [sum(infected[0:iteration_in_day])]
            infected = infected[iteration_in_day:]
        return data

    @staticmethod
    def draw_active_infected(state_changes, plot_days=True, *, plot=plt, title=None, file_path=None):

        active = state_changes[ViralState.INFECTED]
        recovered = state_changes[ViralState.RECOVERED]
        deceased = state_changes[ViralState.DECEASED]

        if plot_days:
            active =  Plotter.to_days(15, active)
            recovered = Plotter.to_days(15, recovered)
            deceased = Plotter.to_days(15, deceased)

        active = np.cumsum(active)
        recovered = np.cumsum(recovered)
        deceased = np.cumsum(deceased)

        active = active - recovered - deceased
        plot.bar(range(len(active)), active)
        plot.bar(range(len(deceased)), deceased, color='red')

        if title is not None:
            plot.set_title(title)

        if file_path is not None:
            plot.savefig(file_path)


    @staticmethod
    def draw_box_plot(data, *, plot=plt, xlabel=None, title=None, ylabel=None, file_path=None):
        n = len(data)
        means, std, minimum, maximum = zip(*data)

        means = np.array(means)
        std = np.array(std)
        minimum = np.array(minimum)
        maximum = np.array(maximum)

        plot.errorbar(np.arange(n), means, std, fmt='ok', lw=3)

        if title is not None:
            plot.set_title(title)

        if xlabel:
            plot.set_xlabel(xlabel)

        if ylabel:
            plot.set_xlabel(ylabel)

        if file_path is not None:
            plot.savefig(file_path)

    @staticmethod
    def draw_infection_heatmap(infected_locations, *, plot=plt, file_path=None):

        x, y = zip(*list(infected_locations.values()))
        plot.scatter(x, y)

        if file_path is not None:
            plot.savefig(file_path)

    @staticmethod
    def draw_bar_graph(data, *, plot=plt, title=None, file_path=None):

        plot.bar(range(len(data)), data)

        if title is not None:
            plot.set_title(title)

        if file_path is not None:
            plot.savefig(file_path)


    @staticmethod
    def draw_agent_movement(agent_paths, *, plot=plt, file_path=None):
        print(agent_paths)
        for path in agent_paths.values():

            x, y = zip(*path)
            plot.plot(x, y)

        if file_path is not None:
            plot.savefig(file_path)
