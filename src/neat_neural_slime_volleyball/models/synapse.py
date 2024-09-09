import numpy as np

class Synapse:
    def __init__(
            self,
            synapse_id: int,
            from_node: int,
            to_node: int
        ):

        self.id = synapse_id
        self.from_node = from_node
        self.to_node = to_node
    
    def _set_weight(self, weight=None, input_size=None, output_size=None):
        if weight is not None:
            self.weight = weight
        else:
            # Scale weights to maintain stable variance (Xavier/Glorot)
            limit = np.sqrt(6 / (input_size + output_size))
            self.weight = np.clip(np.random.uniform(-limit, limit), -1, 1)
    
    def to_dict(self):
        return {
            "id": self.id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "weight": self.weight
        }

    @classmethod
    def from_dict(cls, data):
        synapse = cls(data["id"], data["from_node"], data["to_node"])
        synapse._set_weight(weight=data["weight"])
        return synapse

    def __str__(self):
        return f"Synapse(id={self.id}, from={self.from_node}, to={self.to_node}, w={self.weight:.4f})"

    def __repr__(self):
        return self.__str__()
    