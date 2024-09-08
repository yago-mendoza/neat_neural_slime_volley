import gym
import matplotlib.pyplot as plt
import slimevolleygym
import warnings
from neat_agent import NEATAgent  # We'll implement these in neat_agent.py
from utils import np
from utils.timer import timer
from utils.dual_plot import DualPlot
import random
from datetime import datetime

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym.utils.passive_env_checker")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.utils.passive_env_checker")

# Create the environment
env = gym.make("SlimeVolley-v0") # registered by slimenvollygym when imported

OUTPUT_FOLDER = "output"
timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

# Initialize the NEAT population
population_size = 100 # number of genomes in the population
num_generations = 800 # number of generations to evolve
num_episodes = 3 # number of episodes (games) to evaluate each genome
# (1 game = either player loses 5 lifes (5 rallies) or 3000 steps are reached)
# Create the population of genomes
neat_agent = NEATAgent(
    input_size=env.observation_space.shape[0], # AGENT, BALL & ENEMY (x, y, vx, vy)
    output_size=env.action_space.n, # SLIME'S 3 ACTIONS (MOVE_RIGHT, MOVE_LEFT, JUMP) 
    # Combined actions seem possible: e.g., jump-right [1,0,1] or jump-left [0,1,1]
    population_size=population_size
)

class GenomeStats:
    def __init__(self):
        self.fitness = 0
        self.total_steps = 0
        self.points_scored = 0
        self.points_lost = 0

dual_plot = DualPlot()

# Evolution loop
for generation in range(num_generations):
    print("\n=======================================")
    print(f"> Generation {generation + 1}/{num_generations}")

    data_collection = []

    for i, genome in enumerate(neat_agent.genomes):
        stats = GenomeStats()
        for episode in range(num_episodes):
            obs, done = env.reset(), False
            while not done:
                obs, reward, done, _ = env.step(genome.choose_action(obs))
                stats.fitness += reward
                stats.total_steps += 1
                stats.points_scored += reward > 0
                stats.points_lost += reward < 0
                if np.isscalar(done) and bool(done): break
            
        genome.fitness = stats.fitness # for evolution criteria
        data_collection.append(stats)
        
    scaled_fitnesses = [s.fitness / num_episodes for s in data_collection]
    scaled_steps = [s.total_steps / num_episodes for s in data_collection]

    print("@Population statistics:")
    print(f"  Fitness: μ = {np.mean(scaled_fitnesses):.2f}, [{min(scaled_fitnesses):.2f}, {max(scaled_fitnesses):.2f}]")
    print(f"  Steps:   μ = {np.mean(scaled_steps):.2f}, [{min(scaled_steps):.2f}, {max(scaled_steps):.2f}]")

    best_stats = max(data_collection, key=lambda s: s.fitness)

    best_genome_fitness = best_stats.fitness / num_episodes
    best_genome_steps = best_stats.total_steps / num_episodes

    best_points_scored = best_stats.points_scored / num_episodes
    best_points_lost = best_stats.points_lost / num_episodes

    print("@Best genome:")
    print(f"  Fitness: {best_genome_fitness:.2f}")
    print(f"  Steps:   {best_genome_steps:.2f}")
    print(f"  Points:  {best_points_scored:.2f}/{best_points_lost:.2f}")

    # Update, plot and maybe save
    dual_plot.update(scaled_fitnesses, best_genome_fitness, scaled_steps, best_genome_steps)
    if generation < 45 or random.random() < 0.15:
        dual_plot.plot()
        dual_plot.save(f'{timestamp}.png', OUTPUT_FOLDER)

    # Evolve the population
    print("-------------------")
    print("Evolving genomes...")
    neat_agent.evolve_genomes()

# # Test the best agent
# best_genome = neat_agent.get_best_genome()
# obs, done, total_reward = env.reset(), False, 0

# print("\nTesting best agent:")
# while not done:
#     obs, reward, done, _ = env.step(best_genome.choose_action(obs))
#     total_reward += reward
#     done = done if isinstance(done, bool) else done.item()

# env.close()

# # After testing, print final statistics
# print("\nFinal Statistics:")
# print(f"  Best Fitness: {best_genome.fitness:.2f}")
# print(f"  Final Score: {total_reward:.2f}")

