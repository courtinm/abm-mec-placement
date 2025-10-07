import pygame, sys, random
from simulation.simulator import Simulator
from agents.base_station import BaseStation
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

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    sim = Simulator(grid_size=100)

    sim.add_base_station(BaseStation(1, (10, 10), capacity=50, bs_type="macro"))
    sim.add_base_station(BaseStation(2, (80, 80), capacity=30, bs_type="small"))
    sim.add_base_station(BaseStation(3, (10, 90), capacity=50, bs_type="macro"))  # New Macro BS (bottom-left)
    sim.add_base_station(BaseStation(4, (90, 10), capacity=30, bs_type="small"))  # New Small BS (top-right)


    sim.add_relay_node(RelayNode(1, (30, 30), throughput=30))
    sim.add_relay_node(RelayNode(2, (60, 60), throughput=30))
    #sim.add_relay_node(RelayNode(3, (50, 20), throughput=30))
    #sim.add_relay_node(RelayNode(4, (20, 50), throughput=30))


    for obs in [(45, 45), (50, 50, "large"), (55, 45), (45, 55, "large")]:
        if len(obs) == 2:
            sim.add_obstacle(obs, size='small')
        else:
            sim.add_obstacle((obs[0], obs[1]), size='large')

    for i in range(20):
        sim.add_user(UserDevice(i + 1, (random.randint(0, 100), random.randint(0, 100))))

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
            color = (255, 0, 0) if u.connected_to else (100, 100, 100)
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
