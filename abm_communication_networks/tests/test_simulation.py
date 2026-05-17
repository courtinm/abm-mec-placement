import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.simulator import Simulator
from agents.base_station import BaseStation
from agents.relay_node import RelayNode
from agents.user_device import UserDevice

# Note: These tests focus on core logic and do not cover visualization or randomness aspects.
class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.simulator = Simulator()
        self.base_station = BaseStation(id=1, position=(0, 0), capacity=10)
        self.relay_node = RelayNode(id=1, position=(1, 1), throughput=15)

    def test_base_station_load(self):
        self.base_station.update_load(5)
        self.assertEqual(self.base_station.current_load, 5)
        self.assertFalse(self.base_station.is_overloaded())

        self.base_station.update_load(10)
        self.assertTrue(self.base_station.is_overloaded())

    def test_relay_node_reset(self):
        # RelayNode has current_load and reset(); no update_load/is_overloaded
        self.relay_node.current_load = 4
        self.assertEqual(self.relay_node.current_load, 4)
        self.relay_node.reset()
        self.assertEqual(self.relay_node.current_load, 0)

    def test_add_agents(self):
        self.simulator.add_base_station(self.base_station)
        self.simulator.add_relay_node(self.relay_node)
        self.assertIn(self.base_station, self.simulator.base_stations)
        self.assertIn(self.relay_node, self.simulator.relay_nodes)

    def test_simulate_step_increments_timestep(self):
        self.simulator.add_base_station(self.base_station)
        self.simulator.add_relay_node(self.relay_node)
        self.simulator.add_user(UserDevice(1, (5, 5)))

        self.simulator.simulate_step()
        self.assertEqual(self.simulator.timestep, 1)

    def test_hop_counts_complete(self):
        # Each step must log exactly len(users) hop rows (one per user)
        bs = BaseStation(1, (50, 50), capacity=100, bs_type="macro")
        bs.has_compute_resource = True
        self.simulator.add_base_station(bs)
        self.simulator.add_relay_node(self.relay_node)

        n_users = 5
        for i in range(n_users):
            self.simulator.add_user(UserDevice(i + 1, (50, 50)))

        self.simulator.simulate_step()
        rows_step_1 = [r for r in self.simulator.metrics.hop_counts_log if r[0] == 1]
        self.assertEqual(len(rows_step_1), n_users)


if __name__ == "__main__":
    unittest.main()
