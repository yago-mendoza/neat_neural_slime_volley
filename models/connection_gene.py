class ConnectionGene:
    def __init__(self, in_node, out_node, weight, enabled=True, innovation=None):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation