import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Union, Tuple, Callable
from ..utils.pretty_dict import PrettyDict

from .node import Node
from .synapse import Synapse

class GenomeStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.points_scored = 0
        self.points_lost = 0
        self.total_steps = 0
        self.ball_touches = 0

class Genome:

    def __init__(
        self,
        input_size: int,
        output_size: int
    ):

        self.input_size = input_size
        self.output_size = output_size
        self.nodes: Dict[int, Node] = PrettyDict()
        self.synapses: Dict[int, Synapse] = PrettyDict()

        self.fitness: float = 0
        self.stats = GenomeStats()

        self._init_structure()

    def _init_structure(self):
        self._init_input_layer()
        self._init_output_layer()
        self._input_to_output_batch_synapses()

    def add_hidden_node(
        self,
        from_node: Union[int, Node],
        to_node: Union[int, Node],
        bias: float = 0,
        activation_function: str = "identity"
    ) -> Tuple[int, Node]:
        from_node_id = from_node.id if isinstance(from_node, Node) else from_node
        to_node_id = to_node.id if isinstance(to_node, Node) else to_node

        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError("Invalid node IDs")
        
        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]
        
        if from_node.type == to_node.type in ("input", "output"):
            raise ValueError(f"Cannot add hidden node between 2 {from_node.type} nodes.")
        
        # Remove existing synapse between from_node and to_node
        existing_synapse = next((s for s in self.synapses.values() if s.from_node == from_node_id and s.to_node == to_node_id), None)
        if existing_synapse:
            self._remove_synapse(existing_synapse.id)

        hidden_node_id, hidden_node = self._add_node(type="hidden", bias=bias, activation_function=activation_function)
        self._add_synapse(from_node, hidden_node)
        self._add_synapse(hidden_node, to_node)

        return hidden_node_id, hidden_node
    
    def remove_node(
        self,
        id: int
    ) -> None:
        """
        Remove a node from the genome.

        Args:
            id: The ID of the node to remove.

        Raises:
            ValueError: If the node ID is invalid or if trying to remove an input/output node.
        """

        if id not in self.nodes:
            raise ValueError(f"Node with ID {id} does not exist.")
        
        node = self.nodes[id]

        if node.type in ["input", "output"]:
            raise ValueError(f"Cannot remove {node.type} node with ID {id} since it is part of the I/O structure.")
        
        synapses_to_remove = [
            synapse_id for synapse_id, synapse in self.synapses.items()
            if synapse.from_node == id or synapse.to_node == id
        ]
        
        for synapse_id in synapses_to_remove:
            self._remove_synapse(synapse_id)
        
        del self.nodes[id]
    
    def forward(self, obs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the genome.

        Args:
            obs: The input observations.

        Returns:
            The output values.
        """
        if obs.shape[0] != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {obs.shape[0]}")

        self._set_input_values(obs)
        self._process_nodes()
        return self._get_output_values()

    def _set_input_values(self, obs: np.ndarray):
        for i, value in enumerate(obs):
            self.nodes[i + 1].value = value

    def _process_nodes(self):
        for node in self._topological_sort():
            if node.type != "input":
                node.value = node.activation_function(self._calculate_node_input(node))

    def _calculate_node_input(self, node: Node) -> float:
        return sum(self._get_weighted_input(synapse) for synapse in self._get_incoming_synapses(node)) + node.bias

    def _get_weighted_input(self, synapse: Synapse) -> float:
        return self.nodes[synapse.from_node].value * synapse.weight

    def _get_incoming_synapses(self, node: Node) -> List[Synapse]:
        return [synapse for synapse in self.synapses.values() if synapse.to_node == node.id]

    def _get_output_values(self) -> np.ndarray:
        return np.array([self.nodes[i].value for i in range(self.input_size + 1, self.input_size + self.output_size + 1)])

    def _topological_sort(self) -> List[Node]:
        # Create a graph representation using adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for synapse in self.synapses.values():
            graph[synapse.from_node].append(synapse.to_node)
            in_degree[synapse.to_node] += 1

        # Collect nodes with in-degree 0 (start with input nodes)
        queue = [node for node in self.nodes.values() if in_degree[node.id] == 0]
        sorted_nodes = []

        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)

            for to_node_id in graph[node.id]:
                in_degree[to_node_id] -= 1
                if in_degree[to_node_id] == 0:
                    queue.append(self.nodes[to_node_id])

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Graph has a cycle")

        return sorted_nodes

    def choose_action(self, obs: np.ndarray) -> list:
        output = self.forward(obs)
        # Add small random noise to break ties and increase variation
        output += np.random.normal(0, 1e-3, output.shape)
        return output.tolist()
        
    def _add_node(
        self,
        type: str = "hidden",
        bias: float = 0,
        activation_function: str = "identity"
    ) -> Tuple[int, Node]:

        id = self._find_next_available_id(genre="node")
        node = Node(id=id, type=type).set_bias(bias).set_activation_function(activation_function)
        self.nodes[id] = node
        return id, node
    
    def _add_synapse(
        self,
        from_node: Union[int, Node],
        to_node: Union[int, Node]
    ) -> Tuple[int, Synapse]:

        from_node_id = from_node if isinstance(from_node, int) else from_node.id
        to_node_id = to_node if isinstance(to_node, int) else to_node.id
        
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError("Invalid node IDs")
        
        synapse_id = self._find_next_available_id(genre="synapse")
        synapse = Synapse(synapse_id, from_node_id, to_node_id)
        synapse._set_weight(input_size=self.input_size, output_size=self.output_size)
        self.synapses[synapse_id] = synapse
        return synapse_id, synapse
    
    def _remove_synapse(
        self,
        synapse_id: int
    ) -> None:

        if synapse_id not in self.synapses:
            raise ValueError(f"Synapse with ID {synapse_id} does not exist.")
        
        del self.synapses[synapse_id]

    def _init_input_layer(self):
        for _ in range(self.input_size):
            self._add_node(type="input")

    def _init_output_layer(self):
        for _ in range(self.output_size):
            self._add_node(type="output", activation_function="sigmoid")

    def _input_to_output_batch_synapses(self):

        input_nodes = [n for n in self.nodes.values() if n.type == "input"]
        output_nodes = [n for n in self.nodes.values() if n.type == "output"]
        
        for input_node in input_nodes:
            for output_node in output_nodes:
                self._add_synapse(input_node, output_node)

    def _find_next_available_id(self, genre: str = "node") -> int:
        """Find the smallest positive integer not used as a node or synapse ID."""
        if genre == "node":
            lst = list(self.nodes.keys())
        elif genre == "synapse":
            lst = list(self.synapses.keys())
        else:
            raise ValueError("Invalid genre. Must be 'node' or 'synapse'.")
        return next(i for i in range(1, len(lst) + 2) if i not in set(lst))

    def __str__(self):
        return f"Genome:\nNodes:\n{self.nodes}\nSynapses:\n{self.synapses}"

    def __repr__(self):
        return self.__str__()
    
    def save(self, save_path: str = None):
        """
        Save the genome to a file.

        Args:
            save_path: The path to the file to save the genome to. If None, a timestamped file will be created in the current working directory.

        Example:
            genome.save("genome_20240510_123456_789012.json")
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = f"genome_{timestamp}.json"
        
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        genome_data = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "nodes": {str(node_id): node.to_dict() for node_id, node in self.nodes.items()},
            "synapses": {str(synapse_id): synapse.to_dict() for synapse_id, synapse in self.synapses.items()},
            "fitness": self.fitness
        }

        with open(save_path, "w") as f:
            json.dump(genome_data, f, indent=2)
    
    @classmethod
    def load(cls, load_path: str):
        """
        Load a genome from a file.

        Args:
            load_path: The path to the file to load the genome from.

        Returns:
            The loaded genome.

        Example:
            genome = Genome.load("genome_20240510_123456_789012.json")
        """
        with open(load_path, "r") as f:
            genome_data = json.load(f)

        genome = cls(genome_data["input_size"], genome_data["output_size"])
        genome.nodes = {int(node_id): Node.from_dict(node_data) for node_id, node_data in genome_data["nodes"].items()}
        genome.synapses = {int(synapse_id): Synapse.from_dict(synapse_data) for synapse_id, synapse_data in genome_data["synapses"].items()}
        genome.fitness = genome_data["fitness"]

        return genome