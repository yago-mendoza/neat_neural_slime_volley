class NodeGene:
    def __init__(self, node_id, node_type):
        self.id = node_id
        self.type = node_type  # 'input', 'hidden', or 'output'