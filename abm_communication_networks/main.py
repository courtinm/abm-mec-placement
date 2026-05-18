import random
from simulation.simulator import Simulator
from agents.base_station import BaseStation
from agents.base_station import ComputeRessources
from agents.relay_node import RelayNode
from agents.user_device import UserDevice

# Screen settings 
WIDTH, HEIGHT = 800, 600
BG_COLOR = (30, 30, 40)
FPS = 1

def to_screen(pos):
    x = max(0, min(int(pos[0] / 100 * WIDTH), WIDTH - 1))
    y = max(0, min(int(pos[1] / 100 * HEIGHT), HEIGHT - 1))
    return x, y

def build_simulation(config=None):
    if config is None:
        from configs.default import CONFIG
        config = CONFIG

    sim = Simulator(grid_size=config.get("grid_size", 100))

    for i, bs_cfg in enumerate(config["base_stations"]):
        bs = BaseStation(
            i + 1,
            (bs_cfg["x"], bs_cfg["y"]),
            capacity=bs_cfg.get("capacity", 30),
            bs_type=bs_cfg.get("type", "macro"),
        )
        sim.add_base_station(bs)
        if bs_cfg.get("has_compute_resource", False):
            cr = ComputeRessources(i + 1, bs.position, 10, 0)
            bs.has_compute_resource = True
            bs.compute_resource = cr

    for i, rn_cfg in enumerate(config["relay_nodes"]):
        sim.add_relay_node(RelayNode(i + 1, (rn_cfg["x"], rn_cfg["y"]), throughput=rn_cfg.get("throughput", 30)))

    for obs_cfg in config["obstacles"]:
        sim.add_obstacle((obs_cfg["x"], obs_cfg["y"]), size=obs_cfg.get("size", "small"))

    app_specs = {
        "AR_VR":       {"latency_threshold_ms": 10,  "throughput_req_mbps": 25},
        "streaming":   {"latency_threshold_ms": 50,  "throughput_req_mbps": 10},
        "best_effort": {"latency_threshold_ms": 200, "throughput_req_mbps": 1},
    }
    mix = config.get("app_mix", {"AR_VR": 0.0, "streaming": 0.0, "best_effort": 1.0})
    app_types = list(mix.keys())
    app_weights = list(mix.values())

    for i in range(config["n_users"]):
        app_type = random.choices(app_types, weights=app_weights)[0]
        specs = app_specs[app_type]
        sim.add_user(UserDevice(
            i + 1,
            (random.randint(0, 100), random.randint(0, 100)),
            app_type=app_type,
            **specs,
        ))

    return sim

def main():
    import pygame, sys
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    sim = build_simulation()

    running = True
    while running:
        clock.tick(FPS)
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False

        sim.simulate_step()

        screen.fill(BG_COLOR)

        # Obstacles (color-coded by size)
        for obs in sim.obstacles:
            x, y = to_screen(obs["pos"])
            size = obs.get("size", "small")
            color = (150, 50, 50) if size == "large" else (80, 80, 80)
            pygame.draw.rect(screen, color, (x - 5, y - 5, 10, 10))

        # Base Stations
        for bs in sim.base_stations:
            x, y = to_screen(bs.position)
            color = (0, 0, 255) if bs.bs_type == "macro" else (0, 100, 255)
            radius = 14 if bs.bs_type == "macro" else 8
            if bs.has_compute_resource: #we identify the BS with CR by coloring them in orange
                color = (255, 165, 0)
            pygame.draw.circle(screen, color, (x, y), radius)
            txt = font.render(f"BS{bs.id}:{bs.current_load}", True, (255, 255, 255))
            screen.blit(txt, (x + 10, y - 10))

        # Relay Nodes
        for rn in sim.relay_nodes:
            x, y = to_screen(rn.position)
            pygame.draw.circle(screen, (0, 200, 0), (x, y), 10)
            txt = font.render(f"RN{rn.id}:{rn.current_load}", True, (255, 255, 255))
            screen.blit(txt, (x + 10, y - 10))

            try:
                state = rn.get_state(sim.users)
                q_val = max(rn.agent.q_table.get(state, {0: 0.0}).values())
            except:
                q_val = 0.0

            eps_txt = font.render(f"Q:{q_val:.2f} ε:{rn.agent.epsilon:.2f}", True, (180, 255, 180))
            screen.blit(eps_txt, (x + 10, y + 5))

        # Users
        for u in sim.users:
            x, y = to_screen(u.position)
            color = (255, 0, 0) if u.connected_to else (100, 100, 100) #so every UE is in red if correctly attached
            pygame.draw.circle(screen, color, (x, y), 6)
            txt = font.render(f"U{u.id}", True, (255, 255, 255))
            screen.blit(txt, (x + 5, y + 5))

            if u.connected_to:
                tx, ty = to_screen(u.connected_to.position)
                los_color = (255, 255, 0) if u.has_los else (255, 0, 0)  # Yellow if LOS, Red otherwise
                pygame.draw.line(screen, los_color, (x, y), (tx, ty), 2)

        # Legend
        legend = [("Macro BS", (0, 0, 255)),
                  ("Small BS", (0, 100, 255)),
                  ("RN", (0, 200, 0)),
                  ("User", (255, 0, 0)),
                  ("Lost", (100, 100, 100)),
                  ("Obstacle (small)", (80, 80, 80)),
                  ("Obstacle (large)", (150, 50, 50)),
                  ("LOS (yellow)", (255, 255, 0)),
                  ("NLOS (red)", (255, 0, 0))]

        lx, ly = 10, 10
        for name, col in legend:
            pygame.draw.circle(screen, col, (lx, ly), 6)
            screen.blit(font.render(name, True, (255, 255, 255)), (lx + 15, ly - 6))
            ly += 20

        pygame.display.flip()

    sim.finalize()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
