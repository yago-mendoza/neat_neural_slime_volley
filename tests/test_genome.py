import pytest
import numpy as np
from src.neat_neural_slime_volleyball.models.genome import Genome
from src.neat_neural_slime_volleyball.models.node import Node
from src.neat_neural_slime_volleyball.models.synapse import Synapse

@pytest.fixture
def genome():
    return Genome(input_size=3, output_size=2)

def test_genome_initialization(genome):
    assert len(genome.nodes) == 5  # 3 input + 2 output
    assert len(genome.synapses) == 6  # 3 * 2

def test_node_types(genome):
    input_nodes = [n for n in genome.nodes.values() if n.type == "input"]
    output_nodes = [n for n in genome.nodes.values() if n.type == "output"]
    assert len(input_nodes) == 3
    assert len(output_nodes) == 2

def test_add_hidden_node(genome):
    input_node = next(n for n in genome.nodes.values() if n.type == "input")
    output_node = next(n for n in genome.nodes.values() if n.type == "output")
    hidden_id, hidden_node = genome.add_hidden_node(input_node, output_node)
    assert hidden_node.type == "hidden"
    assert len(genome.nodes) == 6
    assert len(genome.synapses) == 8

def test_add_hidden_node_invalid(genome):
    input_nodes = [n for n in genome.nodes.values() if n.type == "input"]
    with pytest.raises(ValueError):
        genome.add_hidden_node(input_nodes[0], input_nodes[1])

def test_remove_node(genome):
    input_node = next(n for n in genome.nodes.values() if n.type == "input")
    output_node = next(n for n in genome.nodes.values() if n.type == "output")
    hidden_id, _ = genome.add_hidden_node(input_node, output_node)
    initial_synapses = len(genome.synapses)
    genome.remove_node(hidden_id)
    assert len(genome.nodes) == 5
    assert len(genome.synapses) == initial_synapses - 2

def test_remove_input_output_node(genome):
    input_node = next(n for n in genome.nodes.values() if n.type == "input")
    output_node = next(n for n in genome.nodes.values() if n.type == "output")
    with pytest.raises(ValueError):
        genome.remove_node(input_node.id)
    with pytest.raises(ValueError):
        genome.remove_node(output_node.id)

def test_add_synapse(genome):
    input_node = next(n for n in genome.nodes.values() if n.type == "input")
    output_node = next(n for n in genome.nodes.values() if n.type == "output")
    initial_synapses = len(genome.synapses)
    genome._add_synapse(input_node, output_node)
    assert len(genome.synapses) == initial_synapses + 1

def test_remove_synapse(genome):
    synapse_id = next(iter(genome.synapses))
    genome._remove_synapse(synapse_id)
    assert synapse_id not in genome.synapses

def test_remove_nonexistent_synapse(genome):
    with pytest.raises(ValueError):
        genome._remove_synapse(9999)

def test_unique_sequential_ids(genome):
    node_ids = set(genome.nodes.keys())
    assert len(node_ids) == len(genome.nodes)
    assert min(node_ids) == 1
    assert max(node_ids) == len(genome.nodes)

    synapse_ids = set(genome.synapses.keys())
    assert len(synapse_ids) == len(genome.synapses)
    assert min(synapse_ids) == 1
    assert max(synapse_ids) == len(genome.synapses)

def test_weight_initialization(genome):
    for synapse in genome.synapses.values():
        assert isinstance(synapse.weight, float)
        assert -1 <= synapse.weight <= 1

def test_string_representation(genome):
    assert isinstance(str(genome), str)
    assert isinstance(repr(genome), str)

def test_stats_reset(genome):
    genome.stats.points_scored = 10
    genome.stats.points_lost = 5
    genome.stats.total_steps = 100
    genome.stats.ball_touches = 20
    genome.stats.reset()
    assert genome.stats.points_scored == 0
    assert genome.stats.points_lost == 0
    assert genome.stats.total_steps == 0
    assert genome.stats.ball_touches == 0

def test_activation_function_setting(genome):
    node = next(iter(genome.nodes.values()))
    node.set_activation_function('sigmoid')
    assert node.activation_function.__name__ == 'sigmoid'
    node.set_activation_function('relu')
    assert node.activation_function.__name__ == 'relu'

def test_invalid_activation_function(genome):
    node = next(iter(genome.nodes.values()))
    with pytest.raises(KeyError):
        node.set_activation_function('invalid_function')