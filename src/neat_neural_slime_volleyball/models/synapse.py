import numpy as np

class Synapse:
    def __init__(
            self,
            synapse_id: int,
            from_node_id: int,
            to_node_id: int
        ):

        self.id = synapse_id
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
    
    def _set_weight(self, weight=None, input_size=None, output_size=None):
        # Weight is not set as argument as it requires Genome sizes to be set
        if weight is not None:
            self.weight = weight
        else:
            # Scale weights to maintain stable variance (Xavier/Glorot)
            limit = np.sqrt(6 / (input_size + output_size))
            self.weight = np.clip(np.random.uniform(-limit, limit), -1, 1)

    def to_dict(self):
        # This is used to save the genome to a file
        return {
            "id": self.id,
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id   ,
            "weight": self.weight
        }
    @classmethod
    def from_dict(cls, data):
        # This is used to load the genome from a file
        synapse = cls(data["id"], data["from_node_id"], data["to_node_id"])
        synapse._set_weight(weight=data["weight"])
        return synapse

    def __str__(self):
        return f"Synapse(id={self.id}, from_node_id={self.from_node_id}, to_node_id={self.to_node_id}, w={self.weight:.4f})"

    def __repr__(self):
        return self.__str__()
    