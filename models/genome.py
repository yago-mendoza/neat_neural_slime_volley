import numpy as np
from .node_gene import NodeGene
from .connection_gene import ConnectionGene

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