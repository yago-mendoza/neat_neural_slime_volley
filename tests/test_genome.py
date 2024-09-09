import pytest
import sys
import os

from src.neat_neural_slime_volleyball.models.genome import Genome, Node, Connection

@pytest.fixture
def genome():
    return Genome(input_size=3, output_size=2)

def test_genome_initialization(genome):
    assert len(genome.nodes) == 5  # 3 input + 2 output
    assert len(genome.connections) == 6  # 3 * 2

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
    assert len(genome.connections) == 8

def test_add_hidden_node_invalid(genome):
    input_nodes = [n for n in genome.nodes.values() if n.type == "input"]
    with pytest.raises(ValueError):
        genome.add_hidden_node(input_nodes[0], input_nodes[1])

def test_remove_hidden_node(genome):
    input_node = next(n for n in genome.nodes.values() if n.type == "input")
    output_node = next(n for n in genome.nodes.values() if n.type == "output")
    hidden_id, _ = genome.add_hidden_node(input_node, output_node)
    initial_connections = len(genome.connections)
    genome.remove_node(hidden_id)
    assert len(genome.nodes) == 5
    assert len(genome.connections) == initial_connections - 2

def test_remove_input_output_node(genome):
    input_node = next(n for n in genome.nodes.values() if n.type == "input")
    output_node = next(n for n in genome.nodes.values() if n.type == "output")
    with pytest.raises(ValueError):
        genome.remove_node(input_node.id)
    with pytest.raises(ValueError):
        genome.remove_node(output_node.id)

def test_add_connection(genome):
    input_node = next(n for n in genome.nodes.values() if n.type == "input")
    output_node = next(n for n in genome.nodes.values() if n.type == "output")
    initial_connections = len(genome.connections)
    genome._add_connection(input_node, output_node)
    assert len(genome.connections) == initial_connections + 1

def test_remove_connection(genome):
    connection_id = next(iter(genome.connections))
    genome._remove_connection(connection_id)
    assert connection_id not in genome.connections

def test_remove_nonexistent_connection(genome):
    with pytest.raises(ValueError):
        genome._remove_connection(9999)

def test_unique_sequential_ids(genome):
    node_ids = set(genome.nodes.keys())
    assert len(node_ids) == len(genome.nodes)
    assert min(node_ids) == 1
    assert max(node_ids) == len(genome.nodes)

    conn_ids = set(genome.connections.keys())
    assert len(conn_ids) == len(genome.connections)
    assert min(conn_ids) == 1
    assert max(conn_ids) == len(genome.connections)

def test_weight_initialization(genome):
    for conn in genome.connections.values():
        assert isinstance(conn.weight, float)
        assert abs(conn.weight) < 1  # Assuming Xavier/Glorot initialization

def test_bias_initialization(genome):
    for node in genome.nodes.values():
        assert isinstance(node.bias, float)

def test_string_representation(genome):
    assert isinstance(str(genome), str)
    assert isinstance(str(next(iter(genome.nodes.values()))), str)
    assert isinstance(str(next(iter(genome.connections.values()))), str)