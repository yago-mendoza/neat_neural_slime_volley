import numpy as np
import json
import os
import random
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Union, Tuple, Callable
from ..utils.pretty_dict import PrettyDict

from .node import Node, InputNode, OutputNode, HiddenNode
from .synapse import Synapse
import logging

logger = logging.getLogger(__name__)

class GenomeStats:

    def __init__(self):
        self.reset()

    def reset(self):
        self.points_scored = 0
        self.points_lost = 0
        self.total_steps = 0
        self.ball_touches = 0

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

class GenomeHiddenOperationsMixin:

    """

    Initially, all input nodes are fully connected to all output nodes.
    
    From here...

    1. When a hidden node is introduced, the direct synapse between the two nodes it separates is removed.
    2. New synapses will not be created between nodes that are already connected.
    3. The initial structure (fully connected input-output layer) cannot be altered.
    4. If a hidden node is left without any incoming or outgoing synapses, it is removed.

    Ohter rules to ensure mathematical integrity (prohibited feed-back loops,
    disconnected nodes, etc.) are checked by the following methods:

        - def _is_acyclic(self) -> bool
        - def _well_formed(self) -> bool  (basing on _check_integrity() method at 'Node')

    """

    def add_node(
        self,
        from_node: Union[int, Node],
        to_node: Union[int, Node]
    ) -> Tuple[int, Node]:
        """
        Add a hidden node by splitting an existing synapse.
        """
        # Format input & check existance
        from_node = self.nodes[from_node] if isinstance(from_node, int) else from_node
        to_node = self.nodes[to_node] if isinstance(to_node, int) else to_node
        if from_node not in self.nodes.values() or to_node not in self.nodes.values():
            raise ValueError("Invalid node IDs")
        
        existing_opposite_synapse = next(
            (
                s for s in self.synapses.values()
                if s.from_node_id == to_node.id and s.to_node_id == from_node.id
            ),
            None
        )
        if existing_opposite_synapse:
            print(f"Did not apply node because it would create a cycle (genome <{hash(self)}>)")
            # Basically we were trying to set a cycle
            return 0
        
        # [Preparation] Remove existing synapse between from_node and to_node
        existing_synapse = next(
            (
                s for s in self.synapses.values()
                if s.from_node_id == from_node.id and s.to_node_id == to_node.id
            ),
            None
        )
        if existing_synapse:
            del self.synapses[existing_synapse.id]

        hidden_node_id, hidden_node = self._add_node(type="hidden")
        self.add_synapse(from_node, hidden_node, maintenance_mode=True)
        self.add_synapse(hidden_node, to_node, maintenance_mode=True)

        if not self._is_acyclic() or not self._well_formed():
            logger.debug(f"Reverting addition of hidden node {hidden_node_id} due to integrity check failure.")
            self.remove_node(hidden_node_id, maintenance_mode=True)
            if existing_synapse:
                self.add_synapse(from_node, to_node, maintenance_mode=True)
            raise ValueError("Genome became cyclic or malformed after adding a hidden node.")
        
        return hidden_node_id, hidden_node
    
    def remove_node(
        self,
        node: Union[int, Node],
        maintenance_mode: bool = False
    ) -> None:
        # Format input & check existance
        id = node.id if isinstance(node, Node) else node
        node = self.nodes[id]
        if id not in self.nodes:
            raise ValueError(f"Node with ID {id} does not exist.")

        # [!] Cannot remove input or output nodes
        if node.type in ["input", "output"]:
            raise ValueError(f"Cannot remove {node.type} node with ID {id} since it is part of the I/O structure.")

        incoming_nids = [
            synapse.from_node_id for synapse in self.synapses.values()
            if synapse.to_node_id == id
        ]

        outgoing_nids = [
            synapse.to_node_id for synapse in self.synapses.values()
            if synapse.from_node_id == id
        ]

        # [Preparation] Remove all incoming and outgoing synapses before node deletion
        for synapse_id in list(self.synapses.keys()):
            synapse = self.synapses[synapse_id]
            if synapse.from_node_id == id or synapse.to_node_id == id:
                self.remove_synapse(synapse_id, maintenance_mode=True)
                # We don't directly call remove_synapse() because it performs integrity checks
                # that we want to avoid since we are deleting the node anyway.

        # [Operation] Remove node
        del self.nodes[id]
        
        # [Post-operation] Reconnect I/O nodes (if needed)
        # (no needed because of the next operation)
        if not self._is_fully_connected():
            print(f"Re-weaving I/O removal of hidden node (genome <{hash(self)}>)")
            self._weave_input_output_layers()
        # Edge case : in case that the removed node was the last one standing in the
        # middle of 2 I/O nodes. We need to reconnect the I/O nodes inmediatly.

        if not self._is_acyclic() or not self._well_formed():

            print(f"Re-weaving complex removal of hidden node (genome <{hash(self)}>)")

            # If removing the node and its synapses breaks the genome into 2 or more disconnected components
            # we need to reconnect the I/O nodes to the rest of the genome.
            # As long as we cana't know how many were broken, we go all over the neighbors again.

            def revert(a, b):
                if not self._is_acyclic() or not self._well_formed():
                    synapse = self._find_synapse(a, b)
                    self.remove_synapse(synapse, maintenance_mode=True)

            outgoing_nids_copy = outgoing_nids[:]

            for i, node_i in enumerate(incoming_nids):
                if len(outgoing_nids) > 0:
                    node_j = outgoing_nids_copy.pop(0)
                    self.add_synapse(node_i, node_j, maintenance_mode=True)
                    revert(node_i, node_j)
                else:
                    rnode_j = random.choice(outgoing_nids)
                    self.add_synapse(node_i, rnode_j, maintenance_mode=True)
                    revert(node_i, rnode_j)

            if len(outgoing_nids_copy) > 0:
                for node_j in outgoing_nids_copy:
                    rnode_i = random.choice(incoming_nids)
                    self.add_synapse(rnode_i, node_j, maintenance_mode=True)
                    revert(rnode_i, node_j)

            return 1

        return 1

    def add_synapse(
        self,
        from_node: Union[int, Node],
        to_node: Union[int, Node],
        maintenance_mode: bool = False # if True, it will not perform integrity checks
    ) -> Tuple[int, Synapse]:
        # Format input & check existance
        from_node_id = from_node if isinstance(from_node, int) else from_node.id
        to_node_id = to_node if isinstance(to_node, int) else to_node.id
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError("Invalid node IDs")
        
        # [!] Check if synapse already exists
        if self._are_connected(from_node, to_node) or self._are_connected(to_node, from_node):
            for synapse in self.synapses.values():
                if synapse.from_node_id == from_node_id and synapse.to_node_id == to_node_id:
                    return synapse.id, synapse

        # [Operation] Add synapse and set its weight
        synapse_id = self._find_next_available_id(genre="synapse")
        synapse = Synapse(synapse_id, from_node_id, to_node_id)
        synapse._set_weight(input_size=self.input_size, output_size=self.output_size)
        self.synapses[synapse_id] = synapse

        # [Post-!] Check for acyclicity and well-formedness
        if not maintenance_mode:
            if not self._is_acyclic() or not self._well_formed():
                print(f"Reverting addition of synapse (genome <{hash(self)}>)")
                self.remove_synapse(synapse_id, maintenance_mode=True)
                print("Adding this synapse would create an invalid state.")
                return 0

        # [Post-operation] Update node's incoming and outgoing synapses
        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]
        from_node.outgoing_synapses.append(to_node)
        to_node.incoming_synapses.append(from_node)

        return synapse_id, synapse

        # [Post-!] Check for acyclicity and well-formedness
        # if not self._is_acyclic() or not self._well_formed():
        #     self.remove_synapse(synapse_id)  # Rollback if invalid
        #     raise ValueError("Adding this synapse would create an invalid state.")
    
    def remove_synapse(self, synapse: Union[int, Synapse], maintenance_mode: bool = False) -> int:
        """
        Remove a synapse and ensure genome integrity.
        """
        # Format input & check existence
        synapse_id = synapse.id if isinstance(synapse, Synapse) else synapse
        from_node_id = self.synapses[synapse_id].from_node_id
        to_node_id = self.synapses[synapse_id].to_node_id
        if synapse_id not in self.synapses:
            raise ValueError(f"Synapse with ID {synapse_id} does not exist.")

        # [!] Cannot remove a synapse between input and output nodes
        if (from_node_id in self.get_input_layer() and 
            to_node_id in self.get_output_layer()):
            print("Cannot remove synapse between input and output nodes.")
            return 0
        
        # [Operation] Remove synapse
        del self.synapses[synapse_id]

        if not maintenance_mode:
            if not self._is_acyclic() or not self._well_formed():
                logger.debug(f"Reverting removal of synapse {synapse_id} due to integrity check failure.")
                self.add_synapse(from_node_id, to_node_id, maintenance_mode=True)
                raise ValueError("Removing this synapse creates a cycle or disconnects the genome.")
        
        return 1

        # # [Post-operation] Updates node's incoming and outgoing synapses
        # from_node = self.nodes[from_node_id]
        # to_node = self.nodes[to_node_id]    
        # from_node.outgoing_synapses.remove(to_node)
        # to_node.incoming_synapses.remove(from_node)

        return 1

class GenomeForwardPassMixin:

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
        
        if self._is_acyclic() and self._well_formed():
            pass
        else:
            raise ValueError("Genome is not acyclic or well-formed.")

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
        return self.nodes[synapse.from_node_id].value * synapse.weight

    def _get_incoming_synapses(self, node: Node) -> List[Synapse]:
        return [synapse for synapse in self.synapses.values() if synapse.to_node_id == node.id]
    
    def _get_outgoing_synapses(self, node: Node) -> List[Synapse]:
        return [synapse for synapse in self.synapses.values() if synapse.from_node_id == node.id]

    def _get_output_values(self) -> np.ndarray:
        return np.array([self.nodes[i].value for i in range(self.input_size + 1, self.input_size + self.output_size + 1)])

    def _topological_sort(self) -> List[Node]:
        # Create a graph representation using adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for synapse in self.synapses.values():
            graph[synapse.from_node_id].append(synapse.to_node_id)
            in_degree[synapse.to_node_id] += 1

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

class GenomePersistenceMixin:

    def __str__(self):
        nodes = [n for n in list(self.nodes.values()) if n.type == "hidden"]
        synapses = [s for s in list(self.synapses.values()) if s.from_node_id in nodes or s.to_node_id in nodes]
        return f"Genome:\nNodes:\n{nodes}\nSynapses:\n{synapses}"

    def __repr__(self):
        return self.__str__()
    
    def save(self, save_path: str = ""):

        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = f"genome_{timestamp}.json"
        
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        genome_data = self.to_dict()

        with open(save_path, "w") as f:
            json.dump(genome_data, f, indent=2)
    
    def to_dict(self):
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "nodes": {str(node_id): node.to_dict() for node_id, node in self.nodes.items()},
            "synapses": {str(synapse_id): synapse.to_dict() for synapse_id, synapse in self.synapses.items()},
            "fitness": self.fitness
        }
    
    @classmethod
    def load(cls, load_path: Union[str, dict]):
        """
        Load a genome from a file or a dictionary.

        Args:
            load_path: The path to the file to load the genome from or a dictionary containing genome data.

        Returns:
            The loaded genome.

        Example:
            genome = Genome.load("genome_20240510_123456_789012.json")
            genome = Genome.load(genome_data_dict)
        """
        if isinstance(load_path, str):
            with open(load_path, "r") as f:
                genome_data = json.load(f)
        elif isinstance(load_path, dict):
            genome_data = load_path
        else:
            raise ValueError("load_path must be a string (file path) or a dictionary.")

        genome = cls(genome_data["input_size"], genome_data["output_size"])
        genome.nodes = {int(node_id): Node.from_dict(node_data) for node_id, node_data in genome_data["nodes"].items()}
        genome.synapses = {int(synapse_id): Synapse.from_dict(synapse_data) for synapse_id, synapse_data in genome_data["synapses"].items()}
        genome.fitness = genome_data["fitness"]

        return genome

class GenomeRetrievalMixin:
    
    def get_input_layer(self):
        return [node for node in self.nodes.values() if node.type == "input"]
    
    def get_output_layer(self):
        return [node for node in self.nodes.values() if node.type == "output"]
    
    def get_hidden_nodes(self):
        return [node for node in self.nodes.values() if node.type == "hidden"]
    
    def get_hidden_synapses(self):
        return [synapse for synapse in self.synapses.values() if synapse.from_node_id in self.get_hidden_nodes() and synapse.to_node_id in self.get_hidden_nodes()]
    
    def get_directional_data(self, node: Node) -> Dict[str, List[int]]:
        return {
            "incoming": [synapse.from_node_id for synapse in self._get_incoming_synapses(node)],
            "outgoing": [synapse.to_node_id for synapse in self._get_outgoing_synapses(node)]
        }

class Genome(
    GenomeInitializationMixin,
    GenomeHiddenOperationsMixin,
    GenomeForwardPassMixin,
    GenomePersistenceMixin,
    GenomeRetrievalMixin
):
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
    
    def choose_action(self, obs: np.ndarray) -> list:
        output = self.forward(obs)
        # Add small random noise to break ties and increase variation
        output += np.random.normal(0, 1e-3, output.shape)
        return output.tolist()
    

    ### Private methods ###


    def _are_connected(
        self,
        from_node: Union[int, Node],
        to_node: Union[int, Node]
    ) -> bool:
        from_node_id = from_node if isinstance(from_node, int) else from_node.id
        to_node_id = to_node if isinstance(to_node, int) else to_node.id
        return from_node_id in self.nodes and to_node_id in self.nodes and from_node_id in self.nodes[to_node_id].incoming_synapses

    def _is_fully_connected(self):
        # input --> output
        for input_node in self.get_input_layer():
            for output_node in self.get_output_layer():
                if not self._are_connected(input_node, output_node):
                    return False
                
        # input --> hidden
        for input_node in self.get_input_layer():
            for hidden_node in self.get_hidden_nodes():
                if not self._are_connected(input_node, hidden_node):
                    return False
                
        # hidden --> hidden
        for hidden_node in self.get_hidden_nodes():
            for hidden_node in self.get_hidden_nodes():
                if not self._are_connected(hidden_node, hidden_node):
                    return False
        
        # hidden --> output
        for hidden_node in self.get_hidden_nodes():
            for output_node in self.get_output_layer():
                if not self._are_connected(hidden_node, output_node):
                    return False
        
        return True
    
    def _find_next_available_id(self, genre: str = "node") -> int:
        """
        Finds the smallest positive integer not used as a node or synapse ID.
        It is used to assign an ID to a new node or synapse.
        """
        if genre == "node":
            lst = list(self.nodes.keys())
        elif genre == "synapse":
            lst = list(self.synapses.keys())
        else:
            raise ValueError("Invalid genre. Must be 'node' or 'synapse'.")
        return next(i for i in range(1, len(lst) + 2) if i not in set(lst))
    
    def _add_base_synapse(
        self,
        from_node_id: int,
        to_node_id: int
    ) -> Tuple[int, Synapse]:
        """
        Adds a base synapse between I/O nodes at the start of the genome.
        It does not check for acyclicity or well-formedness.
        """
        synapse_id = self._find_next_available_id(genre="synapse")
        synapse = Synapse(synapse_id, from_node_id, to_node_id)
        synapse._set_weight(input_size=self.input_size, output_size=self.output_size)
        self.synapses[synapse_id] = synapse
        self.nodes[from_node_id].outgoing_synapses.append(to_node_id)
        self.nodes[to_node_id].incoming_synapses.append(from_node_id)
        return synapse_id, synapse
    
    def _find_synapse(self, from_node: Union[int, Node], to_node: Union[int, Node]) -> Union[Synapse, None]:
        """
        Encuentra la sinapsis entre dos nodos dados.

        Args:
            from_node: El nodo de origen (puede ser un ID o una instancia de Node).
            to_node: El nodo de destino (puede ser un ID o una instancia de Node).

        Returns:
            La sinapsis que conecta los dos nodos, o None si no se encuentra ninguna sinapsis.
        """
        from_node_id = from_node if isinstance(from_node, int) else from_node.id
        to_node_id = to_node if isinstance(to_node, int) else to_node.id

        for synapse in self.synapses.values():
            if synapse.from_node_id == from_node_id and synapse.to_node_id == to_node_id:
                return synapse

        return None
        
    def _add_node(
        self,
        type: str = "hidden",
    ) -> Tuple[int, Node]:
        """
        To add a new node we only need to define its type.
        The type will trigger default values for 'bias' and 'activation_function'.
        This is set at the 'Node' level.
        """

        id = self._find_next_available_id(genre="node")

        if type == "input":
            node = InputNode(id) # bias = 0, activation_function = 'identity'   
        elif type == "hidden":
            node = HiddenNode(id) # bias = 0, activation_function = 'relu'
        elif type == "output":
            node = OutputNode(id) # bias = 0, activation_function = 'sigmoid' 
        else:
            raise ValueError(f"Invalid node type: {type}")

        self.nodes[id] = node

        return id, node
    
    def _is_acyclic(self) -> bool:
        """Check if the current genome structure is acyclic."""
        try:
            self._topological_sort()
            return True
        except ValueError:
            return False
    
    def _well_formed(self) -> bool:
        """Check if the current genome structure is well-formed."""
        for node in self.nodes.values():
            try:
                node._check_integrity()
            except ValueError as e:
                return False  # Return False if any node fails the integrity check
        return True  # Return True if all nodes pass the integrity check