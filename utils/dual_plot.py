import matplotlib.pyplot as plt
import numpy as np
from utils.timer import timer
from pathlib import Path

class DualPlot:
    def __init__(self, title='Fitness and Steps per Generation', xlabel='Generation', ylabel1='Fitness', ylabel2='Steps'):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        self.fig.suptitle(title)
        
        self.fitness_data = []
        self.fitness_best = []
        self.fitness_mean = []
        
        self.steps_data = []
        self.steps_best = []
        self.steps_mean = []
        
        for ax, ylabel in zip([self.ax1, self.ax2], [ylabel1, ylabel2]):
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', alpha=0.7)

    @timer
    def update(self, fitness_data, fitness_best, steps_data, steps_best):
        self.fitness_data.append(fitness_data)
        self.fitness_best.append(fitness_best)
        self.fitness_mean.append(np.mean(fitness_data))
        
        self.steps_data.append(steps_data)
        self.steps_best.append(steps_best)
        self.steps_mean.append(np.mean(steps_data))

    @timer
    def plot(self):
        for ax, data, best, mean in zip([self.ax1, self.ax2], 
                                        [self.fitness_data, self.steps_data],
                                        [self.fitness_best, self.steps_best],
                                        [self.fitness_mean, self.steps_mean]):
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            num_generations = len(data)
            positions = range(1, num_generations + 1)
            
            if num_generations > 0:
                boxplot = ax.boxplot(data, positions=positions, widths=0.7,
                                     patch_artist=True, showfliers=False)
                
                for box in boxplot['boxes']:
                    box.set(facecolor='mistyrose', alpha=0.8)
                    box.set(edgecolor='none')
                
                for whisker in boxplot['whiskers']:
                    whisker.set(color='lightcoral', alpha=0.3, linewidth=0.5)
                for cap in boxplot['caps']:
                    cap.set(color='lightcoral', alpha=0.3, linewidth=0.5)
                
                for median in boxplot['medians']:
                    median.set(color='red', alpha=0.3, linewidth=0.5)
                
                ax.plot(positions, best, color='red', linewidth=1, label='Best Genome')
                ax.plot(positions, mean, color='green', linestyle='--', linewidth=1, label='Mean')
                
                ax.legend()
                ax.set_xlim(0.5, num_generations + 0.5)  # Adjust x-axis limits
                
                # Dynamically adjust the number of x-axis ticks based on the number of generations
                max_ticks = 20  # Maximum number of ticks to show
                step = max(1, num_generations // max_ticks)
                ticks = range(1, num_generations + 1, step)
                
                ax.set_xticks(ticks)
                ax.set_xticklabels(ticks)
                
                all_values = [item for sublist in data for item in sublist] + best + mean
                y_min, y_max = min(all_values), max(all_values)
                y_range = y_max - y_min
                if y_range == 0:
                    y_range = 1  # Avoid division by zero
                ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
            
            self.ax2.set_xlabel(self.ax2.get_xlabel())
            self.ax1.set_ylabel(self.ax1.get_ylabel())
            self.ax2.set_ylabel(self.ax2.get_ylabel())
        
        plt.tight_layout()

    def show(self):
        plt.tight_layout()
        plt.show()

    @timer
    def save(self, filename, output_folder):
        plt.tight_layout()
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        full_path = output_path / filename
        plt.savefig(full_path, dpi=300, bbox_inches='tight')

    def close(self):
        plt.close(self.fig)