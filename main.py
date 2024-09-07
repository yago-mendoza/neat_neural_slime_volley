import gym
import slimevolleygym
from neat_agent import NEATAgent, evolve_population  # We'll implement these in neat_agent.py

# Create the environment
env = gym.make("SlimeVolley-v0") # registered by slimenvollygym when imported

# Initialize the NEAT population
population_size = 50 # number of genomes in the population
num_generations = 100 # number of generations to evolve
num_episodes = 3 # number of episodes (games) to evaluate each genome
# (1 game = either player loses 5 lifes (5 rallies) or 3000 steps are reached)
# Create the population of genomes
neat_agent = NEATAgent(
    input_size=env.observation_space.shape[0], # AGENT, BALL & ENEMY (x, y, vx, vy)
    output_size=env.action_space.n, # SLIME'S 3 ACTIONS (MOVE_RIGHT, MOVE_LEFT, JUMP) 
    # Combined actions seem possible: e.g., jump-right [1,0,1] or jump-left [0,1,1]
    population_size=population_size
)

# Evolution loop
for generation in range(num_generations):
    print(f"Generation {generation + 1}/{num_generations}")
    
    # Evaluate each genome in the population
    for genome in neat_agent.population:
        fitness = 0
        for _ in range(num_episodes):  # Evaluate each genome over 3 episodes
            obs = env.reset()
            done = False
            while not done:
                action = genome.choose_action(obs)
                obs, reward, done, _ = env.step(action)
                # each action, whether small or large, has an associated reward
                # (even keeping the ball in play if survival_bonus is True)
                fitness += reward 
        genome.fitness = fitness
    
    # Evolve the population
    neat_agent.evolve_population()
    
    # Print best fitness in this generation
    best_fitness = max(genome.fitness for genome in neat_agent.population)
    print(f"Best fitness: {best_fitness}")

# Test the best agent
best_genome = max(neat_agent.population, key=lambda g: g.fitness)
obs = env.reset()
done = False
total_reward = 0

print("\nTesting best agent:")
while not done:
    action = best_genome.choose_action(obs)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()

print(f"Final score: {total_reward}")

env.close()

