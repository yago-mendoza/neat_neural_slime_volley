import pytest
import numpy as np
import os
import tempfile
from src.neat_neural_slime_volleyball.models.genome import Genome

@pytest.fixture
def complex_genome():
    """
    Fixture that creates a complex genome for advanced testing.
    """
    genome = Genome(input_size=4, output_size=3)
    input_node = genome.nodes[1]
    output_node = genome.nodes[5]
    genome.add_hidden_node(input_node, output_node, bias=0.5, activation_function="sigmoid")
    return genome

def test_forward_pass(complex_genome):
    """
    Test the forward method of the genome.

    Ensures:
    1. Output has the correct shape.
    2. Output values are within the expected range.
    3. Different inputs produce different outputs.
    4. Consistent output for the same input.
    """
    input_data = np.array([0.1, 0.2, 0.3, 0.4])
    output1 = complex_genome.forward(input_data)
    
    assert output1.shape == (3,), f"Expected output shape (3,), got {output1.shape}"
    assert np.all((output1 >= 0) & (output1 <= 1)), f"Output values out of range [0, 1]: {output1}"
    
    different_input = np.array([0.5, 0.6, 0.7, 0.8])
    different_output = complex_genome.forward(different_input)
    
    assert not np.array_equal(output1, different_output), "Different inputs produced the same output"
    
    output2 = complex_genome.forward(input_data)
    np.testing.assert_array_almost_equal(output1, output2, decimal=7, err_msg="Inconsistent output for the same input")

def test_choose_action(complex_genome):
    """
    Test the choose_action method of the genome.

    Ensures:
    1. The chosen action is a list of integers.
    2. The action length equals the genome's output size.
    3. Exactly one element of the action is 1 and the rest are 0.
    4. Different inputs can produce different actions.
    """
    input_data = np.array([0.1, 0.2, 0.3, 0.4])
    action = complex_genome.choose_action(input_data)
    
    assert isinstance(action, list), f"Expected list, got {type(action)}"
    assert all(isinstance(x, int) for x in action), f"Not all elements are integers: {action}"
    assert len(action) == complex_genome.output_size, f"Expected length {complex_genome.output_size}, got {len(action)}"
    assert sum(action) == 1, f"Sum of action should be 1, got {sum(action)}"
    assert set(action) == {0, 1}, f"Action should only contain 0 and 1, got {set(action)}"

    # Test for different actions with different inputs
    actions = [complex_genome.choose_action(np.random.rand(4)) for _ in range(10)]
    assert len(set(tuple(a) for a in actions)) > 1, "Different inputs should be capable of producing different actions"

def test_save_and_load(complex_genome):
    """
    Test the save and load methods of the genome.

    Ensures:
    1. Genome can be saved to a file.
    2. Genome can be loaded from the file.
    3. Loaded genome has the same structure as the original.
    4. Loaded genome produces identical output to the original.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
        temp_path = tmp_file.name
    
    try:
        complex_genome.save(temp_path)
        assert os.path.exists(temp_path), f"File not created at {temp_path}"
        
        loaded_genome = Genome.load(temp_path)
        
        assert loaded_genome.input_size == complex_genome.input_size, "Input size mismatch"
        assert loaded_genome.output_size == complex_genome.output_size, "Output size mismatch"
        assert len(loaded_genome.nodes) == len(complex_genome.nodes), "Node count mismatch"
        assert len(loaded_genome.synapses) == len(complex_genome.synapses), "Synapse count mismatch"
        
        for node_id, node in complex_genome.nodes.items():
            loaded_node = loaded_genome.nodes[node_id]
            assert node.to_dict() == loaded_node.to_dict(), f"Node mismatch for id {node_id}"
        
        for synapse_id, synapse in complex_genome.synapses.items():
            loaded_synapse = loaded_genome.synapses[synapse_id]
            assert synapse.to_dict() == loaded_synapse.to_dict(), f"Synapse mismatch for id {synapse_id}"
        
        input_data = np.array([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_almost_equal(
            loaded_genome.forward(input_data),
            complex_genome.forward(input_data),
            decimal=7,
            err_msg="Forward pass mismatch between original and loaded genome"
        )
    
    finally:
        os.remove(temp_path)

def test_topological_sort(complex_genome):
    """
    Test the _topological_sort method of the genome.

    Ensures:
    1. All nodes are present in the topological order.
    2. Input nodes appear first.
    3. Output nodes appear last.
    4. Hidden nodes are in between.
    5. For each synapse, the source node appears before the destination node.
    """
    sorted_nodes = complex_genome._topological_sort()
    
    assert len(sorted_nodes) == len(complex_genome.nodes), "Not all nodes are in the sorted list"
    
    input_nodes = [n for n in sorted_nodes if n.type == "input"]
    output_nodes = [n for n in sorted_nodes if n.type == "output"]
    hidden_nodes = [n for n in sorted_nodes if n.type == "hidden"]
    
    assert all(sorted_nodes.index(n) < len(input_nodes) for n in input_nodes), "Input nodes are not at the beginning"
    assert all(sorted_nodes.index(n) >= len(sorted_nodes) - len(output_nodes) for n in output_nodes), "Output nodes are not at the end"
    assert all(len(input_nodes) <= sorted_nodes.index(n) < len(sorted_nodes) - len(output_nodes) for n in hidden_nodes), "Hidden nodes are not in the middle"
    
    for synapse in complex_genome.synapses.values():
        from_index = next(i for i, n in enumerate(sorted_nodes) if n.id == synapse.from_node)
        to_index = next(i for i, n in enumerate(sorted_nodes) if n.id == synapse.to_node)
        assert from_index < to_index, f"Synapse {synapse.id} violates topological order"

def test_genome_stats(complex_genome):
    """
    Test the functionality of genome statistics.

    Ensures:
    1. Statistics can be updated correctly.
    2. The reset method of statistics works as expected.
    """
    complex_genome.stats.points_scored = 5
    complex_genome.stats.points_lost = 3
    complex_genome.stats.total_steps = 100
    complex_genome.stats.ball_touches = 20
    
    assert complex_genome.stats.points_scored == 5, "Points scored not set correctly"
    assert complex_genome.stats.points_lost == 3, "Points lost not set correctly"
    assert complex_genome.stats.total_steps == 100, "Total steps not set correctly"
    assert complex_genome.stats.ball_touches == 20, "Ball touches not set correctly"
    
    complex_genome.stats.reset()
    
    assert complex_genome.stats.points_scored == 0, "Points scored not reset"
    assert complex_genome.stats.points_lost == 0, "Points lost not reset"
    assert complex_genome.stats.total_steps == 0, "Total steps not reset"
    assert complex_genome.stats.ball_touches == 0, "Ball touches not reset"

def test_invalid_forward_input(complex_genome):
    """
    Test that the forward method raises an exception when given invalid input.

    Ensures:
    1. ValueError is raised when input size doesn't match input_size.
    2. The error message is descriptive.
    """
    with pytest.raises(ValueError) as excinfo:
        complex_genome.forward(np.array([0.1, 0.2, 0.3]))
    assert "Expected 4 inputs, got 3" in str(excinfo.value), "Incorrect error message for invalid input"

def test_cycle_detection(complex_genome):
    """
    Test that the _topological_sort method detects cycles in the genome graph.

    Ensures:
    1. ValueError is raised when a cycle is introduced in the graph.
    2. The error message is descriptive.
    """
    output_node = next(n for n in complex_genome.nodes.values() if n.type == "output")
    input_node = next(n for n in complex_genome.nodes.values() if n.type == "input")
    complex_genome._add_synapse(output_node, input_node)
    
    with pytest.raises(ValueError) as excinfo:
        complex_genome._topological_sort()
    assert "Graph has a cycle" in str(excinfo.value), "Cycle not detected or incorrect error message"

def test_activation_functions(complex_genome):
    """
    Test different activation functions for nodes.

    Ensures:
    1. Different activation functions can be set and used.
    2. Invalid activation functions raise appropriate errors.
    """
    node = next(iter(complex_genome.nodes.values()))
    
    for func in ['sigmoid', 'tanh', 'relu']:
        node.set_activation_function(func)
        assert node.activation_function.__name__ == func, f"Activation function not set to {func}"
    
    with pytest.raises(KeyError) as excinfo:
        node.set_activation_function('invalid_function')
    assert "invalid_function" in str(excinfo.value), "Incorrect error message for invalid activation function"

def test_add_remove_node(complex_genome):
    """
    Test adding and removing nodes from the genome.

    Ensures:
    1. Hidden nodes can be added between existing nodes.
    2. Nodes can be removed (except input/output nodes).
    3. Appropriate errors are raised for invalid operations.
    """
    def test_add_remove_node(complex_genome):
        initial_node_count = len(complex_genome.nodes)
        initial_synapse_count = len(complex_genome.synapses)
        
        input_node = next(n for n in complex_genome.nodes.values() if n.type == "input")
        output_node = next(n for n in complex_genome.nodes.values() if n.type == "output")
        
        hidden_id, hidden_node = complex_genome.add_hidden_node(input_node, output_node)
        assert len(complex_genome.nodes) == initial_node_count + 1, "Node not added"
        assert len(complex_genome.synapses) == initial_synapse_count + 1, "Synapses not updated correctly after adding node"
        
        complex_genome.remove_node(hidden_id)
        assert len(complex_genome.nodes) == initial_node_count, "Node not removed"
        assert len(complex_genome.synapses) == initial_synapse_count, "Synapses not updated correctly after removing node"
        
        with pytest.raises(ValueError) as excinfo:
            complex_genome.remove_node(input_node.id)
        assert "Cannot remove input node" in str(excinfo.value), "Incorrect error message for removing input node"

def test_fitness_update(complex_genome):
    """
    Test updating and retrieving the fitness of the genome.

    Ensures:
    1. Fitness can be set and retrieved correctly.
    2. Fitness is preserved when saving and loading the genome.
    """
    complex_genome.fitness = 0.75
    assert complex_genome.fitness == 0.75, "Fitness not set correctly"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
        temp_path = tmp_file.name
    
    try:
        complex_genome.save(temp_path)
        loaded_genome = Genome.load(temp_path)
        assert loaded_genome.fitness == 0.75, "Fitness not preserved after save and load"
    
    finally:
        os.remove(temp_path)
