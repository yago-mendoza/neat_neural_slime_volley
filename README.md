![Example GIF](https://github.com/yago-mendoza/neat_neural_slime_volley/blob/main/assets/header_giff.gif)

# NEAT Neural Slime Volleyball

This project implements a NeuroEvolution of Augmenting Topologies (NEAT) algorithm to evolve neural networks for playing a simplified version of volleyball, known as Slime Volleyball.

## Project Overview

The project is structured into several modules, including agents, models, and utilities, each serving a specific purpose in the NEAT algorithm. The core of the project lies in the Genome class, which represents the neural network as a sequence of nodes and synapses. The genome undergoes various operations such as crossover, mutation, and fitness evaluation to evolve over generations.

## Genome Class

The Genome class is designed to manage the neural network's structure and parameters. It:

- Initializes with a fully connected input-output layer
- Allows for the addition and removal of hidden nodes and synapses
- Ensures mathematical integrity by checking for acyclicity and well-formedness
- Handles crossover by combining genes from two parent genomes
- Includes error handling to manage inconsistencies during crossover
- Evaluates fitness based on performance in the Slime Volleyball environment

## Project Structure

The project is organized into several directories:

- `agents`: Contains the NEATAgent class, which manages the population of genomes and handles the evolution process.
- `models`: Contains the Genome, Node, and Synapse classes, which represent the neural network's structure.
- `utils`: Contains utility functions for simulation, plotting, and genome visualization.
- `vendor`: Contains external dependencies and fixed versions of the Slime Volleyball environment.

## Key Features

- **NeuroEvolution**: Implements the NEAT algorithm to evolve neural networks.
- **Genome Management**: Handles the initialization, mutation, and crossover of genomes.
- **Fitness Evaluation**: Evaluates the performance of genomes in the Slime Volleyball environment.
- **Visualization**: Provides tools for visualizing the genome structure and performance.

## Dependencies

- numpy: For numerical operations.
- pytest: For testing.
- imageio: For rendering visualizations.
- Custom modules for simulation, plotting, and genome visualization.

## Example Code References

### Crossover Process and Error Handling

The following code snippet from `src/neat_neural_slime_volleyball/agents/neat_agent.py` demonstrates the crossover process and error handling:

```python
for innovation_number in all_innovation_numbers:
    try:
        if innovation_number in parent1_genes and innovation_number in parent2_genes:
            # Matching genes: randomly choose one parent's version
            chosen_synapse = random.choice([parent1_genes[innovation_number], parent2_genes[innovation_number]])
        elif innovation_number in parent1_genes:
            # Disjoint or excess gene from parent1
            chosen_synapse = parent1_genes[innovation_number] if parent1 == more_fit_parent else None
        else:
            # Disjoint or excess gene from parent2
            chosen_synapse = parent2_genes[innovation_number] if parent2 == more_fit_parent else None

        if chosen_synapse:
            # Ensure nodes are added before their synapse
            if chosen_synapse.from_node_id not in child.nodes:
                child.nodes[chosen_synapse.from_node_id] = deepcopy(parent1.nodes[chosen_synapse.from_node_id])
            if chosen_synapse.to_node_id not in child.nodes:
                child.nodes[chosen_synapse.to_node_id] = deepcopy(parent1.nodes[chosen_synapse.to_node_id])
            child.synapses[chosen_synapse.id] = deepcopy(chosen_synapse)
    except KeyError as e:
        print(f"Warning: Skipping synapse with innovation number {innovation_number} due to missing node: {e}")
```

### Genome Structure Initialization

The following code snippet from `src/neat_neural_slime_volleyball/models/genome.py` shows the initialization and management of the genome structure:

```python
class GenomeInitializationMixin:
    def _init_structure(self):
        self._init_input_layer()
        self._init_output_layer()
        self._weave_input_output_layers()

    def _init_input_layer(self):
        for _ in range(self.input_size):
            self._add_node(type="input")

    def _init_output_layer(self):
        for _ in range(self.output_size):
            self._add_node(type="output")

    def _weave_input_output_layers(self):
        input_nodes = [self.nodes[i] for i in range(1, self.input_size + 1)]
        output_nodes = [self.nodes[i] for i in range(self.input_size + 1, self.input_size + self.output_size + 1)]
        for input_node in input_nodes:
            for output_node in output_nodes:
                try:
                    self._add_base_synapse(input_node.id, output_node.id)
                except ValueError as e:
                    print(f"Error adding base synapse between {input_node.id} and {output_node.id}: {e}")
```

### Synapse Class Definition

The following code snippet from `src/neat_neural_slime_volleyball/models/synapse.py` shows the definition and methods of the Synapse class:

```python
import numpy as np

class Synapse:
    def __init__(
        self,
        synapse_id: int,
        from_node_id: int,
        to_node_id: int,
        weight: float = 0.0,
        enabled: bool = True
    ):
        self.id = synapse_id
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.weight = weight
        self.enabled = enabled

    def _set_weight(self, weight=None, input_size=None, output_size=None):
        if weight is not None:
            self.weight = weight
        else:
            limit = np.sqrt(6 / (input_size + output_size))
            self.weight = np.clip(np.random.uniform(-limit, limit), -1, 1)

    def to_dict(self):
        return {
            "id": self.id,
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "weight": self.weight,
            "enabled": self.enabled
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            synapse_id=data["id"],
            from_node_id=data["from_node_id"],
            to_node_id=data["to_node_id"],
            weight=data["weight"],
            enabled=data["enabled"]
        )

    def __str__(self):
        return f"Synapse(id={self.id}, from_node_id={self.from_node_id}, to_node_id={self.to_node_id}, w={self.weight:.4f})"

    def __repr__(self):
        return self.__str__()
```
