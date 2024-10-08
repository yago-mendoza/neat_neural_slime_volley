import pygame
import json
import time
import math
from ..models.genome import Genome

class GenomeVisualizer:

    """
    A class to visualize a genome.

    # Usage:
        # gvis = GenomeVisualizer()
        # gvis.visualize('g3.json')
    """

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Genome Visualization")
        self.clock = pygame.time.Clock()
        self.nodes = {}
        self.connections = []
        self.dragging_node = None
        self.selected_nodes = set()
        self.background_surface = pygame.Surface((800, 600))

    def load_genome(self, genome_json_path):
        with open(genome_json_path, 'r') as f:
            genome_data = json.load(f)
        self.genome = Genome.load(genome_data)
        self.setup_visualization()

    def setup_visualization(self):
        self.nodes.clear()
        self.connections.clear()

        input_nodes = [node_id for node_id in self.genome.nodes if self.genome.nodes[node_id].type == "input"]
        output_nodes = [node_id for node_id in self.genome.nodes if self.genome.nodes[node_id].type == "output"]
        hidden_nodes = [node_id for node_id in self.genome.nodes if self.genome.nodes[node_id].type == "hidden"]

        # Calculate spacing based on the number of nodes
        input_spacing = (600 - 100) / max(1, len(input_nodes))  # Space for input nodes
        output_spacing = (600 - 100) / max(1, len(output_nodes))  # Space for output nodes
        hidden_spacing = (600 - 100) / max(1, len(hidden_nodes))  # Space for hidden nodes

        # Adjusted radius for nodes
        node_radius = 16  # Smaller radius for nodes

        # Position input nodes
        for i, node_id in enumerate(input_nodes):
            self.nodes[node_id] = {'pos': (50, 50 + i * input_spacing), 'radius': node_radius, 'color': (173, 216, 230)}

        # Position output nodes
        for i, node_id in enumerate(output_nodes):
            self.nodes[node_id] = {'pos': (700, 50 + i * output_spacing), 'radius': node_radius, 'color': (255, 200, 200)}  # Lighter gray

        # Position hidden nodes
        for i, node_id in enumerate(hidden_nodes):
            self.nodes[node_id] = {'pos': (375, 50 + i * hidden_spacing), 'radius': node_radius, 'color': (0, 0, 0)}

        for synapse in self.genome.synapses.values():
            self.connections.append((synapse.from_node_id, synapse.to_node_id))

        self.draw_background()  # Draw the static background once

    def draw_background(self):
        self.background_surface.fill((255, 255, 255))  # Clear background

        for from_node, to_node in self.connections:
            from_node_type = self.genome.nodes[from_node].type
            to_node_type = self.genome.nodes[to_node].type

            # Skip connections from input nodes to output nodes
            if from_node_type == "input" and to_node_type == "output":
                continue

            from_pos = self.nodes[from_node]['pos']
            to_pos = self.nodes[to_node]['pos']
            color = (150, 150, 150) if from_node_type == "hidden" or to_node_type == "hidden" else (240, 240, 240)  # Lighter gray

            # Draw the line
            pygame.draw.line(self.background_surface, color, from_pos, to_pos, 2)

            # Draw the arrow in the middle of the line
            self.draw_arrow(from_pos, to_pos, color)
        
        for node_id, node in self.nodes.items():
            pos = node['pos']
            pygame.draw.circle(self.background_surface, node['color'], pos, node['radius'])  # Draw node
            font = pygame.font.Font(None, 36)
            text = font.render(str(node_id), True, (255, 255, 255))
            self.background_surface.blit(text, (pos[0] - 10, pos[1] - 10))  # Draw node ID


    def visualize(self, genome_json_path=None, genome=None):

        if genome:
            self.genome = genome
            self.setup_visualization()
        else:
            self.load_genome(genome_json_path)

        running = True
        while running:
            self.handle_events()  # Handle events like window close
            self.draw()           # Draw the current genome
            self.clock.tick(60)   # Limit to 60 FPS

            if self.window_closed():  # Check if the window is closed
                running = False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.handle_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    self.handle_mouse_up(event)
            elif event.type == pygame.MOUSEMOTION:
                self.handle_mouse_motion(event)
    
    def draw_arrow(self, start_pos, end_pos, color):
        """
        Draw an arrow in the middle of the line from start_pos to end_pos with the given color.
        """
        arrow_size = 10  # Size of the arrow head
        arrow_angle = 30  # Angle of the arrow head

        # Calculate the middle point of the line
        mid_pos = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)

        # Calculate the direction vector
        direction = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
        length = (direction[0]**2 + direction[1]**2) ** 0.5
        direction = (direction[0] / length, direction[1] / length)

        # Calculate the points for the arrow head
        left_point = (
            mid_pos[0] - arrow_size * (direction[0] * math.cos(math.radians(arrow_angle)) - direction[1] * math.sin(math.radians(arrow_angle))),
            mid_pos[1] - arrow_size * (direction[1] * math.cos(math.radians(arrow_angle)) + direction[0] * math.sin(math.radians(arrow_angle)))
        )
        right_point = (
            mid_pos[0] - arrow_size * (direction[0] * math.cos(math.radians(-arrow_angle)) - direction[1] * math.sin(math.radians(-arrow_angle))),
            mid_pos[1] - arrow_size * (direction[1] * math.cos(math.radians(-arrow_angle)) + direction[0] * math.sin(math.radians(-arrow_angle)))
        )

        # Draw the arrow head
        pygame.draw.polygon(self.background_surface, color, [mid_pos, left_point, right_point])  # Use a bright color for visibility

    def draw(self):
        self.screen.blit(self.background_surface, (0, 0))  # Draw the static background

        # Draw the dragging node with a thin red outline
        if self.dragging_node is not None:
            pos = self.nodes[self.dragging_node]['pos']
            pygame.draw.circle(self.screen, (255, 0, 0), pos, self.nodes[self.dragging_node]['radius'], 1)  # Thin red outline

        pygame.display.flip()  # Update the display

    def handle_mouse_down(self, event):
        # Check if clicking on a node
        for node_id, node in self.nodes.items():
            pos = node['pos']
            if (event.pos[0] - pos[0]) ** 2 + (event.pos[1] - pos[1]) ** 2 < node['radius'] ** 2:
                # Deselect other nodes
                self.selected_nodes.clear()
                self.dragging_node = node_id  # Set the dragging node
                return

    def handle_mouse_up(self, event):
        if self.dragging_node is not None:
            self.dragging_node = None  # Stop dragging
            self.draw_background()  # Redraw background after moving

    def handle_mouse_motion(self, event):
        if self.dragging_node is not None:
            self.nodes[self.dragging_node]['pos'] = event.pos  # Update position of the dragged node

    def window_closed(self):
        return pygame.event.get(pygame.QUIT)