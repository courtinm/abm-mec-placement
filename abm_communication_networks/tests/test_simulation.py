import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.simulator import Simulator
from agents.base_station import BaseStation
from agents.relay_node import RelayNode

class TestSimulation(unittest.TestCase):
    def setUp(self):
        """Set up the simulator and agents for testing."""
        self.simulator = Simulator()
        self.base_station = BaseStation(id=1, position=(0, 0), capacity=10)
        self.relay_node = RelayNode(id=1, position=(1, 1), throughput=15)

    def test_base_station_load(self):
        """Test updating load on a base station."""
        self.base_station.update_load(5)
        self.assertEqual(self.base_station.current_load, 5)
        self.assertFalse(self.base_station.is_overloaded())

        self.base_station.update_load(10)
        self.assertTrue(self.base_station.is_overloaded())

    def test_relay_node_load(self):
        """Test updating load on a relay node."""
        self.relay_node.update_load(10)
        self.assertEqual(self.relay_node.current_load, 10)
        self.assertFalse(self.relay_node.is_overloaded())

        self.relay_node.update_load(6)
        self.assertTrue(self.relay_node.is_overloaded())

    def test_add_agents(self):
        """Test adding base stations and relay nodes to the simulator."""
        self.simulator.add_base_station(self.base_station)
        self.simulator.add_relay_node(self.relay_node)

        self.assertIn(self.base_station, self.simulator.base_stations)
        self.assertIn(self.relay_node, self.simulator.relay_nodes)

    def test_simulate_network(self):
        """Test the simulate_network method for agent interactions."""
        self.simulator.add_base_station(self.base_station)
        self.simulator.add_relay_node(self.relay_node)

        initial_position_base_station = self.base_station.position
        initial_position_relay_node = self.relay_node.position

        self.simulator.simulate_network()

        # Check that positions are updated
        self.assertNotEqual(self.base_station.position, initial_position_base_station)
        self.assertNotEqual(self.relay_node.position, initial_position_relay_node)

        # Check that loads are updated
        self.assertGreaterEqual(self.base_station.current_load, 1)
        self.assertGreaterEqual(self.relay_node.current_load, 1)

    def test_print_network_status(self):
        """Test printing the network status."""
        self.simulator.add_base_station(self.base_station)
        self.simulator.add_relay_node(self.relay_node)

        # Run simulation to update states
        self.simulator.simulate_network()

        # Ensure print_network_status executes without errors
        try:
            self.simulator.print_network_status()
        except Exception as e:
            self.fail(f"print_network_status raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()
