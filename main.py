import warnings
from datetime import datetime
import time
import random
import imageio
import os

from src.neat_neural_slime_volleyball.vendor.fixed_slimevolley import SlimeVolleyEnv
from src.neat_neural_slime_volleyball.utils.simulation import Simulation
from src.neat_neural_slime_volleyball.agents.neat_agent import NEATAgent
from src.neat_neural_slime_volleyball.vendor import np
from src.neat_neural_slime_volleyball.utils.timer import timer
from src.neat_neural_slime_volleyball.utils.dual_plot import DualPlot
from src.neat_neural_slime_volleyball.utils.genome_visualizer import GenomeVisualizer

# ------------------------
# Configuration Parameters 
# ------------------------

# Simulation Parameters
population_size = 100 # Number of genomes in the population
num_generations = 100 # Number of generations to evolve the population
num_episodes = 3 # Number of episodes to run for each genome

# Mutation Parameters
mr_synapse = 0.20 # Mutation rate for synapse weights
mr_bias = 0.10 # Mutation rate for biases
mr_activation = 0.10 # Mutation rate for activation functions
glob_mr = 1.00 # Global mutation rate multiplier

# Topology mutation switch weights
w_ahn = 0.75 # Weight of add-hidden-node mutation
w_rhn = 0.35 # Weight of remove-hidden-node mutation
w_ahs = 0.50 # Weight of add-hidden-synapse mutation
w_rhs = 0.0 # Weight of remove-hidden-synapse mutation

# Plotting Parameters
DO_PLOT = True # Plot the fitness and steps of genomes across generations
DO_JSON = True # Save the best genome as a JSON file

OUTPUT_FOLDER = "output" # Folder for output PNG files (from DualPlot, if DO_PLOT=True)
SESSION_NAME = datetime.now().strftime("%Y.%m.%d.%H.%M.%S") # Rewritten at each generation

# Rendering Parameters
RENDER_BEST = True # Render the best genome after evolution
DRAW_BEST = True # Draw the best genome after evolution

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

    if DO_PLOT: dual_plot = DualPlot()

    BEST_GENOME = neat_agent.genomes[0]

    # [2] Evolution loop
    for generation in range(num_generations):
        print(f"\n##########")
        print(f"> G-({generation + 1}/{num_generations})")
        print(f"Starting simulation with {len(neat_agent.genomes)} genomes... (render=False)")

        for genome in neat_agent.genomes:

            try:

                # [2.1] Run simulation
                sim = Simulation(env, genome, num_episodes, render=False)
                sim.run() # stat collection at genome
            
                # [2.2] Calculate fitness
                neat_agent.calculate_fitness(genome) # fitness collection at genome
            
            except Exception as e:

                vg = GenomeVisualizer()
                print(hash(genome))
                # vg.visualize(genome=genome)

                print(f"Error running simulation: {e}")
                print(f"| WARNING: Genome is not acyclic or well-formed.")
                print(f"| SYS: Saved the current genome structure as 'output/{SESSION_NAME}/flawed_acyclic_genome.json'")
                genome.save(f'output/{SESSION_NAME}/flawed_acyclic_genome.json')

        # [2.3] Display inline results

        best_genome = max(neat_agent.genomes, key=lambda g: g.fitness)

        fs_lst, ss_lst =_print_population_performance_statistics(neat_agent)
        _print_best_genome_performance_statistics(best_genome)

        print(f"-------------------")

        # [2.4] Update saved plot
        if DO_PLOT:
            dual_plot.update(fs_lst, best_genome.fitness, ss_lst, best_genome.stats.total_steps)
            dual_plot.plot()
            dual_plot.save_plot(f'plot.png', f"{OUTPUT_FOLDER}/{SESSION_NAME}")
        
        # [2.5] Save best genome JSON
        if DO_JSON:
            best_genome_json_path = f"{OUTPUT_FOLDER}/{SESSION_NAME}/best_genomes/bg{generation + 1}n{len(best_genome.get_hidden_nodes()):03d}.json"
            best_genome.save(best_genome_json_path)
        
        if best_genome.fitness > BEST_GENOME.fitness:
            BEST_GENOME = best_genome

        # [2.6] Evolve the population
        
        # try:

        mc = neat_agent.evolve_genomes(
            mr_synapse=mr_synapse,
            mr_bias=mr_bias,
            mr_activation=mr_activation,
            glob_mr=glob_mr,
            w_ahn=w_ahn,
            w_rhn=w_rhn,
            w_ahs=w_ahs,
            w_rhs=w_rhs
        ) # Get mutation count
        print(f"Population evolved.")
        print(f"-------------------")

        # [3.1] Print mutation count
        _print_mutation_count(mc)

        # [3.2] Print parametric statistics
        _print_parametric_population_statistics(neat_agent)

        # [3.3] Print topological statistics
        _print_topological_population_statistics(neat_agent)

        # [3.4] Print best genome parametric and topological statistics
        _print_parametric_and_topological_genome_statistics(best_genome)

        # except Exception as e:
        #     print(f"Error running evolution: {e}")
        #     print(f"| WARNING: Genome is not acyclic or well-formed.")
        #     print(f"| SYS: Saved the current genome structure as 'output/{SESSION_NAME}/flawed_acyclic_genome.json'")
        #     genome.save(f'output/{SESSION_NAME}/flawed_acyclic_genome.json')
        

    # [4] Render the best genome game
    if RENDER_BEST:
        print("-------------------")
        print("Preparing the final simulation...")
        input(">> Press ENTER to render.")
        Simulation(env, BEST_GENOME).run()

    # [5] Draw the best genome game
    if DRAW_BEST:
        print("-------------------")
        print("Drawing the best genome game...")
        gvis = GenomeVisualizer()

        best_genome_json_path = f"{OUTPUT_FOLDER}/{SESSION_NAME}/best_genome.json"
        BEST_GENOME.save(best_genome_json_path)
        
        gvis.visualize(best_genome_json_path)




# Data printing functions

def _print_population_performance_statistics(neat_agent):
    print(f"@Population performance:")
    fitnesses = [genome.fitness for genome in neat_agent.genomes]
    steps = [genome.stats.total_steps for genome in neat_agent.genomes]
    print(f"      Fitness: μ = {np.mean(fitnesses):.2f}, [{min(fitnesses):.2f}, {max(fitnesses):.2f}]")
    print(f"  [!] Steps:   μ = {np.mean(steps):.2f}, [{min(steps):.2f}, {max(steps):.2f}]")
    return fitnesses, steps

def _print_best_genome_performance_statistics(best_genome):
    print(f"@Best genome performance:")
    print(f"  [!] Fitness: {best_genome.fitness:.2f}")
    print(f"      Steps:   {best_genome.stats.total_steps:.2f}")
    print(f"  [!] Points:  {best_genome.stats.points_scored:.2f}-{best_genome.stats.points_lost:.2f}")

def _print_mutation_count(mc):
    total_mc = sum(mc.values())
    mutation_strings = []
    for mutation_type in ['ahn', 'rhn', 'ahs', 'rhs']:
        mutation_percentage = mc[mutation_type] / total_mc * 100
        mutation_strings.append(f"{mutation_type.upper()}: {mutation_percentage:.2f}%")
    mutation_summary = ", ".join(mutation_strings)
    print(f"@Mutation count ({total_mc}): {mutation_summary}")

def _print_parametric_population_statistics(neat_agent):
    print(f"@Parametric statistics:")

    biases = []
    weights = []
    for genome in neat_agent.genomes:
        biases.append([node.bias for node in genome.nodes.values()])
        weights.append([synapse.weight for synapse in genome.synapses.values()])

    if biases:
        mean_bias = np.mean([np.mean(bs) for bs in biases]) 
        min_bias = np.min([np.min(bs) for bs in biases])
        max_bias = np.max([np.max(bs) for bs in biases])
        print(f"  Bias: μ = {mean_bias:.4f} [{min_bias:.4f}, {max_bias:.4f}]")
    else:
        print("  No biases found in hidden nodes.")
    
    if weights:
        mean_weight = np.mean([np.mean(ws) for ws in weights])
        min_weight = np.min([np.min(ws) for ws in weights])
        max_weight = np.max([np.max(ws) for ws in weights])
        print(f"  Weight: μ = {mean_weight:.4f} [{min_weight:.4f}, {max_weight:.4f}]")
    else:
        print("  No weights found in hidden synapses.")

def _print_topological_population_statistics(neat_agent):
    print(f"@Topological statistics:")

    hidden_nodes_count = [len(genome.get_hidden_nodes()) for genome in neat_agent.genomes]

    mean_hidden_nodes = np.mean(hidden_nodes_count)
    max_hidden_nodes = np.max(hidden_nodes_count)
    min_hidden_nodes = np.min(hidden_nodes_count)

    print(f"  Hidden nodes: μ = {mean_hidden_nodes} [{min_hidden_nodes}, {max_hidden_nodes}]")
    
    hidden_incoming_synapses_count = []
    for genome in neat_agent.genomes:
        for node in genome.get_hidden_nodes():
            hidden_incoming_synapses_count.append(len(node.incoming_synapses))

    hidden_outgoing_synapses_count = []
    for genome in neat_agent.genomes:
        for node in genome.get_hidden_nodes():
            hidden_outgoing_synapses_count.append(len(node.outgoing_synapses))

    mean_hidden_incoming_synapses = np.mean(hidden_incoming_synapses_count)
    max_hidden_incoming_synapses = np.max(hidden_incoming_synapses_count)
    min_hidden_incoming_synapses = np.min(hidden_incoming_synapses_count)

    mean_hidden_outgoing_synapses = np.mean(hidden_outgoing_synapses_count)
    max_hidden_outgoing_synapses = np.max(hidden_outgoing_synapses_count)
    min_hidden_outgoing_synapses = np.min(hidden_outgoing_synapses_count)

    print(f"  Hidden incoming synapses: μ = {mean_hidden_incoming_synapses} [{min_hidden_incoming_synapses}, {max_hidden_incoming_synapses}]")
    print(f"  Hidden outgoing synapses: μ = {mean_hidden_outgoing_synapses} [{min_hidden_outgoing_synapses}, {max_hidden_outgoing_synapses}]")
    
    activation_counts = {
        'identity': 0,
        'sigmoid': 0,
        'tanh': 0,
        'relu': 0
    }

    total_hidden_nodes = sum(len(genome.get_hidden_nodes()) for genome in neat_agent.genomes)
    
    for genome in neat_agent.genomes:
        for node in genome.get_hidden_nodes():
            activation = node.activation_function.__name__
            activation_counts[activation] = activation_counts.get(activation, 0) + 1

    for key in activation_counts:
        activation_counts[key] = round((activation_counts[key] / total_hidden_nodes) * 100, 2)

    print(f"  Activation functions at hidden nodes (I-S-T-R)[%] : {(activation_counts['identity'], activation_counts['sigmoid'], activation_counts['tanh'], activation_counts['relu'])}")

def _print_parametric_and_topological_genome_statistics(best_genome):
    print(f"@Best genome (parametric+topological):")
        
    hidden_nodes_count = len(best_genome.get_hidden_nodes())
    print(f"  Hidden nodes: {len(best_genome.get_hidden_nodes())}")

    if hidden_nodes_count > 0:

        print(f"  Hidden-hidden synapses: {len(best_genome.get_hidden_synapses())}")

        activation_counts = {
                'identity': 0,
                'sigmoid': 0,
                'tanh': 0,
                'relu': 0
            }
        for node in best_genome.get_hidden_nodes():
            activation = node.activation_function.__name__  # Get the string name of the activation function
            activation_counts[activation] = activation_counts.get(activation, 0) + 1
        
        print(f"  Activation functions at hidden nodes (I-S-T-R) : {(activation_counts['identity'], activation_counts['sigmoid'], activation_counts['tanh'], activation_counts['relu'])}")

    else:
        print(f"  Hidden synapses: None")
        print(f"  Activation functions: None")

if __name__ == "__main__":
    main()
