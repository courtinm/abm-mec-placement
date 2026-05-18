import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import build_simulation


def _run(config, n_steps, output_dir, seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    sim = build_simulation(config)
    for _ in range(n_steps):
        sim.simulate_step()
    sim.finalize(output_dir=str(output_dir))


def test_simulation_runs(tmp_path):
    random.seed(0)
    sim = build_simulation()
    for _ in range(10):
        sim.simulate_step()
    sim.finalize(output_dir=str(tmp_path))


def test_seeding_is_deterministic(tmp_path):
    from configs.rural import CONFIG

    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    dir_a.mkdir()
    dir_b.mkdir()

    _run(CONFIG, 20, dir_a, seed=42)
    _run(CONFIG, 20, dir_b, seed=42)

    assert (dir_a / "hop_counts.csv").read_text() == (dir_b / "hop_counts.csv").read_text()
