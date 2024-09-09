import pygame
import sys
from vendor.fixed_slimevolley import SlimeVolleyEnv
import numpy as np
from typing import Tuple, List, Optional
from neat_agent import Genome

class Simulation:
    """
    Handles the simulation of the SlimeVolley environment for a given genome.
    This class is used in main.py to evaluate genomes in the NEAT algorithm.
    """

    def __init__(self, env, genome, num_episodes=1, collect_data=True, render=True):
        """
        Initialize the Simulation.

        :param env: The SlimeVolleyEnv instance.
        :param genome: The Genome instance to be evaluated.
        :param num_episodes: Number of episodes to run for this simulation.
        :param collect_data: Whether to collect data at genome statistics.
        :param render: Whether to render the simulation visually.
        """
        self.env = env
        self.genome = genome
        self.num_episodes = num_episodes
        self.COLLECT_DATA = collect_data
        self.RENDER = render
        
        if self.RENDER: pygame.init()
        if self.RENDER: self.clock = pygame.time.Clock()
        
        # Some genomes may have survived from previous generations.
        self.genome.stats.reset()
        # We want to evaluate their performance fresh

        # Used to track ball touches
        self.prev_ball_x = None
        self.prev_ball_y = None

    def run(self):
        """Run the simulation for the specified number of episodes."""
        for _ in range(self.num_episodes):
            self._run_episode()

    def _run_episode(self):
        """Run a single episode of the simulation."""
        max_frames = 3000
        obs, done = self.env.reset(), False
        frame_count = 0
        score_left, score_right = 0, 0
        
        if self.RENDER: self.env.render()
        
        while not done:
            frame_count += 1
            
            if self.RENDER and pygame.event.get(pygame.QUIT): break
            
            # Use the genome to choose an action based on the current observation
            action = self.genome.choose_action(obs)
            obs, reward, done, _ = self.env.step(action)
            
            if self.COLLECT_DATA:
                self.update_genome_stats(obs, reward)
            
            if self.RENDER:
                self.env.render()
                pygame.display.flip()
                self.clock.tick(30)  # Limit to 30 FPS
                if reward != 0:
                    score_right += int(reward > 0)
                    score_left += int(reward < 0)
                    print(self.format_progress(frame_count, max_frames, score_left, score_right))
        if self.RENDER:
            print(">> Game simulation complete.")
            self.env.close()
            pygame.quit()
    @staticmethod
    def format_progress(frame, total, score_left, score_right, bar_length=50):
        filled_length = int(bar_length * frame // total)
        bar = '=' * filled_length + '>' + '.' * (bar_length - filled_length - 1)
        return f"Progress: [{bar}] {frame:4d}/{total} | Score: {score_left}-{score_right}"
    
    def update_genome_stats(self, obs, reward):
        self.genome.stats.total_steps += 1 / self.num_episodes
        self.genome.stats.points_scored += max(reward, 0) / self.num_episodes
        self.genome.stats.points_lost += max(-reward, 0) / self.num_episodes
        
        ball_x, ball_y = obs[4], obs[5]
        if self.prev_ball_x is not None and self.prev_ball_y is not None:
            if abs(ball_x - self.prev_ball_x) > 0.1 or abs(ball_y - self.prev_ball_y) > 0.1:
                self.genome.stats.ball_touches += 1 / self.num_episodes
        self.prev_ball_x, self.prev_ball_y = ball_x, ball_y





        
