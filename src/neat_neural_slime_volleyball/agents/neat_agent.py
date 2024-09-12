from ..vendor import np
from ..utils.timer import timer
from ..models.genome import Genome
from ..models.node import Node
from functools import partial
from copy import deepcopy

class NEATAgent:
    def __init__(self, input_size, output_size, population_size=50):
        self.input_size = input_size # 13
        self.output_size = output_size # 3
        self.population_size = population_size
        self.genomes = [
            Genome(input_size, output_size)
            for _ in range(population_size)
        ]
    
    def calculate_fitness(self, genome: Genome):

        stats = genome.stats

        # Normalize values
        max_steps = 3000  # Maximum possible steps in a game
        max_touches = 100  # Arbitrary max, adjust based on observations

        normalized_points = (stats.points_scored - stats.points_lost + 5) / 10  # Shift to 0-1 range
        normalized_steps = stats.total_steps / max_steps
        normalized_touches = min(stats.ball_touches / max_touches, 1)

        # Weight the components
        point_weight = 0.75
        step_weight = 0.25
        touch_weight = 0.00

        fitness = (
            point_weight * normalized_points +
            step_weight * normalized_steps +
            touch_weight * normalized_touches
        )

        genome.fitness = fitness
        return fitness
    
    def parametric_mutation(
        self,
        genome: Genome,
        mr_synapse: float,
        mr_bias: float,
        mr_activation: float
    ):
        """Mutate the genome by adjusting weights or activation functions."""

        for synapse in genome.synapses.values():
            if np.random.rand() < mr_synapse:  # Adjust weight with 10% probability
                synapse.weight += np.random.uniform(-mr_synapse, mr_synapse)

        for node in genome.nodes.values():
            if np.random.rand() < mr_bias:  # Adjust weight with 10% probability
                node.bias += np.random.uniform(-mr_bias, mr_bias)
        
        for node in genome.get_hidden_nodes():
            if np.random.rand() < mr_activation:  # Adjust activation function with 10% probability
                node.rswitch_activation_function()
            if np.random.rand() < mr_activation:  # Adjust bias with 10% probability
                node.bias += np.random.uniform(-mr_activation, mr_activation)

    def topological_mutation(
        self,
        g: Genome,
        w_ahn: float,
        w_rhn: float,
        w_ahs: float,
        w_rhs: float,
        m = None
    ):
        """Mutate the genome by adding/removing nodes or synapses."""

        if len(g.nodes) == g.input_size + g.output_size:
            m = 'ahn'

        if not m:
            p = [w_ahn, w_rhn, w_ahs, w_rhs]
            m = np.random.choice(
                [
                    'ahn',  # Add hidden node
                    'rhn',  # Remove hidden node
                    'ahs',  # Add hidden synapse
                    'rhs'   # Remove hidden synapse
                ], 
                # we normalize it
                p=[x/sum(p) for x in p]
            )

        INS = g.get_input_layer()
        HNS = g.get_hidden_nodes()
        ONS = g.get_output_layer()

        # Create a partial function with the current parameters
        partial_topological_mutation = partial(
            self.topological_mutation,
            w_ahn=w_ahn,
            w_rhn=w_rhn,
            w_ahs=w_ahs,
            w_rhs=w_rhs
        )

        if m == 'ahn':
            from_node, to_node = None, None
            while from_node == to_node :
                from_node = np.random.choice(INS + HNS)
                to_node = np.random.choice(HNS + ONS)
            assert from_node.type != 'output'
            assert to_node.type != 'input'
            g.add_node(from_node, to_node)
            return m
        
        elif m == 'rhn':
            if len(HNS) > 1:
                node_to_remove = np.random.choice(HNS)
                g.remove_node(node_to_remove)
                return m
            else:
                # Accelerate things a bit
                return partial_topological_mutation(g=g, m='ahn')
        
        elif m == 'ahs':
            if len(HNS) > 0: # There is something to connect
                if g._is_fully_connected(): # del: there will always be room
                    partial_topological_mutation(g=g, m='ahn')
                else:
                    from_node, to_node = None, None
                    while True:
                        from_node = np.random.choice(INS + HNS)
                        to_node = np.random.choice(ONS + HNS)
                        if from_node != to_node and not g._are_connected(from_node, to_node):
                            break
                    
                    out = g.add_synapse(from_node, to_node)
                    if not out:
                        return partial_topological_mutation(g=g, m='ahn')
                    else:
                        return m
            else:
                return partial_topological_mutation(g=g, m='ahn')

        if m == 'rhs':
            # Get synapses that go from one hidden neuron to another hidden neuron
            hidden_synapses = [
                s for s in g.synapses.values()
                if s.from_node_id in HNS and s.to_node_id in HNS
            ]
            
            if hidden_synapses:
                # Randomly select a hidden synapse to remove
                synapse_to_remove = np.random.choice(hidden_synapses)
                out = g.remove_synapse(synapse_to_remove.id)
                if not out:
                    return partial_topological_mutation(g=g, m='ahs')
                else:
                    return m
            else:
                # If there are no hidden synapses, you could try another mutation
                return partial_topological_mutation(g=g, m='ahs')
        
        
            

    # def crossover(
    #     self,
    #     parent1: Genome,
    #     parent2: Genome
    # ):
    #     child =  Genome(self.input_size, self.output_size)
    #     return child

    def crossover(
        self,
        parent1: Genome,
        parent2: Genome
    ):
        # Create a deep copy of parent1
        child = deepcopy(parent1)


        #Idea, empieza desde los nodos INPUT
        # primero sclo a ppel

        return child

    @timer
    def evolve_genomes(
        self,
        mr_synapse: float = 0.35,
        mr_bias: float = 0.15,
        mr_activation: float = 0.10,
        glob_mr: float = 1,
        w_ahn: float = 0.75,
        w_rhn: float = 0.0,
        w_ahs: float = 0.50,
        w_rhs: float = 0.0
    ):
        if glob_mr != 1:
            mr_synapse *= glob_mr
            mr_bias *= glob_mr
            mr_activation *= glob_mr

        print(f"Evolving genomes...")

        # Sort genomes by fitness
        self.genomes.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep the top half
        top_half = self.genomes[:self.population_size // 2]

        mutation_count = {
            'ahn': 0,
            'rhn': 0,
            'ahs': 0,
            'rhs': 0
        }
        
        # Create new genomes to replace the bottom half
        new_genomes = []
        for _ in range(self.population_size - len(top_half)):
            parent1, parent2 = np.random.choice(top_half, 2, replace=False)
            child = self.crossover(parent1, parent2)

            self.parametric_mutation(child, mr_synapse, mr_bias, mr_activation) 
            m = self.topological_mutation(child, w_ahn, w_rhn, w_ahs, w_rhs)

            new_genomes.append(child)

            mutation_count[m] += 1
        
        for _ in range(len(top_half)):
            self.parametric_mutation(top_half[_], mr_synapse, mr_bias, mr_activation)
            m = self.topological_mutation(top_half[_], w_ahn, w_rhn, w_ahs, w_rhs)
            mutation_count[m] += 1
        
        self.genomes = top_half + new_genomes

        return mutation_count
