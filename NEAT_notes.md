# NEAT - NeuroEvolution of Augmenting Topologies

- **NN** = Topology + Weights
- **Neuro-evolutionary algorithm** (incremental grow from minimal structure; first GA able to optimize and complexify solutions simultaneously)

1. Create a population of random NNs
2. Evaluate the fitness of each NN
3. Select the fittest NNs (as prents for next generation)
4. Breed the fittest NNs (crossover /+ mutation -> offspring)
5. Repeat

## Terminology

- **Genome** = NN (phenotype) as a sequence of connections (node -> node)
    - Innovation numbers solve the Competing Conventions Problem
- **Population** = population of NNs
- **Species** = population of NNs that have similar topology (NNs compete among similars)
- **Bloat** = tendency to increase the size of the NN without improving the fitness (overfitting)
    - Simpler NNs get eliminated before they can develop into efficient solutions
    - Complex NNs able to explore the solution-space by brute-force are kept
- **Complexification** = NNs are allowed to grow only if they are more fit
- **Speciation** = NNs breed with others that are similar to them, avoiding being eliminated
    - Protects structural innovation
- **Fitness sharing** = NNs within a species share the fitness reward, encouraging diversity
- **Pruning** = removing excess connections from the NN

## Insights

- Avg. Hidden Neurons vs Nº of Species decrease
    - Means NEAT is favoring simpler solutions over time

- **Backprop NEAT** = NEAT (evolve topologies) + Backpropagation (evolve weights)
    - (use backpropagation to update the weights of the NN and further finetune it)
    - Supervised learning (data separation) -> Backprop NEAT
    - Reinforcement learning (no explicit "right answer") -> NEAT