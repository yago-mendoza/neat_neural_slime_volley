import warnings
from datetime import datetime
import time
import pygame
import random

from src.neat_neural_slime_volleyball.vendor.fixed_slimevolley import SlimeVolleyEnv
from src.neat_neural_slime_volleyball.utils.simulation import Simulation
from src.neat_neural_slime_volleyball.agents.neat_agent import NEATAgent
from src.neat_neural_slime_volleyball.vendor import np
from src.neat_neural_slime_volleyball.utils.timer import timer
from src.neat_neural_slime_volleyball.utils.dual_plot import DualPlot


# ------------------------
# Configuration Parameters 
# ------------------------

# Simulation Parameters
population_size = 100 # Number of genomes in the population
num_generations = 100 # Number of generations to evolve the population
num_episodes = 2 # Number of episodes to run for each genome

# Plotting Parameters
DO_PLOT = False # Plot the fitness and steps of genomes across generations
OUTPUT_FOLDER = "output" # Folder for output PNG files (from DualPlot, if DO_PLOT=True)
plot_filename = datetime.now().strftime("%Y.%m.%d.%H.%M.%S") # Rewritten at each generation

# Rendering Parameters
RENDER_BEST = False # Render the best genome after evolution


########################

def main():
        
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="gym.utils.passive_env_checker")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.utils.passive_env_checker")

    env = SlimeVolleyEnv() # Fixed version from the official lib

    # [1] Create the population of genomes
    neat_agent = NEATAgent(
        input_size=env.observation_space.shape[0],
        output_size=env.action_space.n,
        # todo: combined actions seem possible: e.g., jump-right [1,0,1] or jump-left [0,1,1]
        population_size=population_size
    )

    # [2] Evolution loop
    for generation in range(num_generations):
        print(f"> Generation {generation + 1}/{num_generations}")

        for genome in neat_agent.genomes:

            # [2.1] Run simulation
            sim = Simulation(env, genome, num_episodes, render=False)
            sim.run() # stat collection at genome

            # [2.2] Calculate fitness
            neat_agent.calculate_fitness(genome) # fitness collection at genome

        # [2.3] Display inline results
        print("@Population statistics:")
        fitnesses = [genome.fitness for genome in neat_agent.genomes]
        steps = [genome.stats.total_steps for genome in neat_agent.genomes]
        print(f"  Fitness: μ = {np.mean(fitnesses):.2f}, [{min(fitnesses):.2f}, {max(fitnesses):.2f}]")
        print(f"  Steps:   μ = {np.mean(steps):.2f}, [{min(steps):.2f}, {max(steps):.2f}]")
        print("@Best genome:")
        best_genome = max(neat_agent.genomes, key=lambda g: g.fitness)
        print(f"  Fitness: {best_genome.fitness:.2f}")
        print(f"  Steps:   {best_genome.stats.total_steps:.2f}")
        print(f"  Points:  {best_genome.stats.points_scored:.2f}-{best_genome.stats.points_lost:.2f}")

        # [2.4] Update saved plot
        if DO_PLOT:
            dual_plot = DualPlot()
            dual_plot.update(fitnesses, best_genome.fitness, steps, best_genome.stats.total_steps)
            dual_plot.plot()
            dual_plot.save(f'{plot_filename}.png', OUTPUT_FOLDER)

        # [2.5] Evolve the population
        print("-------------------")
        print("Evolving genomes...")
        neat_agent.evolve_genomes()

    # [3] Render the best genome
    if RENDER_BEST:
        Simulation(env, best_genome).run()

if __name__ == "__main__":
    main()
