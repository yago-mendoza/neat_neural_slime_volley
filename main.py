
from vendor.fixed_slimevolley import SlimeVolleyEnv
from utils.sym_render import simulate_game
import warnings
from neat_agent import NEATAgent
from vendor import np
from utils.timer import timer
from utils.dual_plot import DualPlot
from datetime import datetime
import time
import pygame
import random

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym.utils.passive_env_checker")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.utils.passive_env_checker")

env = SlimeVolleyEnv() 

OUTPUT_FOLDER = "output"
timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

# Initialize the NEAT population
population_size = 100 # number of genomes in the population
num_generations = 100 # number of generations to evolve
num_episodes = 2 # number of episodes (games) to evaluate each genome
# (1 game = either player loses 5 lifes (5 rallies) or 3000 steps are reached)

# Create the population of genomes
neat_agent = NEATAgent(
    input_size=env.observation_space.shape[0], # AGENT, BALL & ENEMY (x, y, vx, vy)
    output_size=env.action_space.n, # SLIME'S 3 ACTIONS (MOVE_RIGHT, MOVE_LEFT, JUMP) 
    # Combined actions seem possible: e.g., jump-right [1,0,1] or jump-left [0,1,1]
    population_size=population_size
)

dual_plot = DualPlot()

# Evolution loop
for generation in range(num_generations):
    print("\n=======================================")
    print(f"> Generation {generation + 1}/{num_generations}")

    for genome in neat_agent.genomes:
        genome.stats.reset()  # Reset stats for each genome before evaluation

        for episode in range(num_episodes):
            obs, done = env.reset(), False
            prev_ball_x, prev_ball_y = obs[4], obs[5]  # Initial ball position
            while not done:
                action = genome.choose_action(obs)
                obs, reward, done, _ = env.step(action)
                
                genome.stats.total_steps += 1 / num_episodes # total steps taken
                genome.stats.points_scored += max(reward, 0) / num_episodes # points scored
                genome.stats.points_lost += max(-reward, 0) / num_episodes # points lost
                
                # Check for ball touch
                ball_x, ball_y = obs[4], obs[5]
                if abs(ball_x - prev_ball_x) > 0.1 or abs(ball_y - prev_ball_y) > 0.1:
                    genome.stats.ball_touches += 1
                prev_ball_x, prev_ball_y = ball_x, ball_y
                
                if np.isscalar(done) and bool(done): break
        
        # Calculate fitness after all episodes
        genome.fitness = neat_agent.calculate_fitness(genome.stats)
        
    print("@Population statistics:")

    fitnesses = [genome.fitness for genome in neat_agent.genomes]
    steps = [genome.stats.total_steps for genome in neat_agent.genomes]

    print(f"  Fitness: μ = {np.mean(fitnesses):.2f}, [{min(fitnesses):.2f}, {max(fitnesses):.2f}]")
    print(f"  Steps:   μ = {np.mean(steps):.2f}, [{min(steps):.2f}, {max(steps):.2f}]")

    print("@Best genome:")

    best_genome = max(neat_agent.genomes, key=lambda g: g.fitness)

    print(f"  Fitness: {best_genome.fitness:.2f}")
    print(f"  Steps:   {best_genome.stats.total_steps:.2f}")
    print(f"  Points:  {best_genome.stats.points_scored:.2f}/{best_genome.stats.points_lost:.2f}")

    dual_plot.update(fitnesses, best_genome.fitness, steps, best_genome.stats.total_steps)
    dual_plot.plot()
    dual_plot.save(f'{timestamp}.png', OUTPUT_FOLDER)

    # Evolve the population
    print("-------------------")
    print("Evolving genomes...")
    neat_agent.evolve_genomes()

simulate_game(env, best_genome)
