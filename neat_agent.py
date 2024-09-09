from vendor import np
from utils.timer import timer

class NodeGene:
    def __init__(self, node_id, node_type):
        self.id = node_id
        self.type = node_type  # 'input', 'hidden', or 'output'

class ConnectionGene:
    def __init__(self, in_node, out_node, weight, enabled=True, innovation=None):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

class GenomeStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.points_scored = 0
        self.points_lost = 0
        self.total_steps = 0
        self.ball_touches = 0

class Genome:
    def __init__(self, input_size: int, output_size: int):
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.weights: np.ndarray = np.random.randn(input_size, output_size)
        self.bias: np.ndarray = np.random.randn(output_size)

        self.fitness: float = 0
        self.stats = GenomeStats()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.weights) + self.bias

    def choose_action(self, obs: np.ndarray) -> list:
        output = self.forward(obs)
        action = (output == np.max(output)).astype(int)
        return action.tolist() 

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
        point_weight = 0.9
        step_weight = 0.05
        touch_weight = 0.05

        fitness = (
            point_weight * normalized_points +
            step_weight * normalized_steps +
            touch_weight * normalized_touches
        )

        genome.fitness = fitness

        return fitness

    @timer
    def evolve_genomes(self):
        # Sort genomes by fitness
        self.genomes.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep the top half
        top_half = self.genomes[:self.population_size // 2]
        
        # Create new genomes to replace the bottom half
        new_genomes = []
        for _ in range(self.population_size - len(top_half)):
            parent = np.random.choice(top_half)
            child = Genome(self.input_size, self.output_size)
            child.weights = parent.weights + np.random.normal(0, 0.1, parent.weights.shape)
            child.bias = parent.bias + np.random.normal(0, 0.1, parent.bias.shape)
            new_genomes.append(child)
        
        self.genomes = top_half + new_genomes