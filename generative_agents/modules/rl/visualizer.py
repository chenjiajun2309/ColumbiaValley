"""
RL Metrics Visualizer

Generate visualizations and reports for RL training metrics.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional
from collections import defaultdict


class RLMetricsVisualizer:
    """
    Generate visualizations for RL training metrics
    """
    
    def __init__(self, metrics_file: str = None, output_dir: str = None):
        """
        Initialize visualizer
        
        Args:
            metrics_file: Path to metrics JSON file
            output_dir: Directory to save visualizations
        """
        self.metrics_file = metrics_file
        self.output_dir = output_dir or "."
        self.metrics_data = None
        
        if metrics_file and os.path.exists(metrics_file):
            self.load_metrics(metrics_file)
    
    def load_metrics(self, filepath: str):
        """Load metrics from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            self.metrics_data = json.load(f)
    
    def plot_reward_trends(self, save_path: str = None):
        """
        Plot reward trends for each agent over training steps
        
        Args:
            save_path: Path to save the figure
        """
        if not self.metrics_data:
            return
        
        reward_history = self.metrics_data.get("reward_history", {})
        if not reward_history:
            return
        
        fig, axes = plt.subplots(len(reward_history), 1, figsize=(12, 4 * len(reward_history)))
        if len(reward_history) == 1:
            axes = [axes]
        
        for idx, (agent_name, history) in enumerate(reward_history.items()):
            if not history:
                continue
            
            steps = [r["step"] for r in history]
            rewards = [r["reward"] for r in history]
            
            axes[idx].plot(steps, rewards, alpha=0.6, linewidth=1, label="Reward")
            
            # Add moving average
            if len(rewards) > 5:
                window = min(10, len(rewards) // 5)
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                moving_steps = steps[window-1:]
                axes[idx].plot(moving_steps, moving_avg, linewidth=2, label="Moving Average", color='red')
            
            axes[idx].set_xlabel("Training Step")
            axes[idx].set_ylabel("Reward")
            axes[idx].set_title(f"{agent_name} - Reward Trend")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, "reward_trends.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def plot_reward_components(self, save_path: str = None):
        """
        Plot reward components for each agent
        
        Args:
            save_path: Path to save the figure
        """
        if not self.metrics_data:
            return None
        
        reward_components = self.metrics_data.get("reward_components", {})
        if not reward_components:
            return None
        
        # Check if we have any data
        has_data = False
        for agent_name, components in reward_components.items():
            if components:
                for component, values in components.items():
                    if values and len(values) > 0:
                        has_data = True
                        break
                if has_data:
                    break
        
        if not has_data:
            return None
        
        fig, axes = plt.subplots(len(reward_components), 1, figsize=(12, 4 * len(reward_components)))
        if len(reward_components) == 1:
            axes = [axes]
        
        for idx, (agent_name, components) in enumerate(reward_components.items()):
            if not components:
                continue
            
            # Plot each component
            for component, values in components.items():
                if not values:
                    continue
                steps = [v["step"] for v in values]
                component_values = [v["value"] for v in values]
                axes[idx].plot(steps, component_values, alpha=0.6, label=component, linewidth=1.5)
            
            axes[idx].set_xlabel("Training Step")
            axes[idx].set_ylabel("Component Value")
            axes[idx].set_title(f"{agent_name} - Reward Components")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, "reward_components.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def plot_reward_distribution(self, save_path: str = None):
        """
        Plot reward distribution for each agent
        
        Args:
            save_path: Path to save the figure
        """
        if not self.metrics_data:
            return None
        
        reward_history = self.metrics_data.get("reward_history", {})
        if not reward_history:
            return None
        
        # Check if we have any data
        has_data = False
        for agent_name, history in reward_history.items():
            if history and len(history) > 0:
                has_data = True
                break
        
        if not has_data:
            return None
        
        num_agents = len(reward_history)
        fig, axes = plt.subplots(1, num_agents, figsize=(6 * num_agents, 5))
        if num_agents == 1:
            axes = [axes]
        
        for idx, (agent_name, history) in enumerate(reward_history.items()):
            if not history:
                continue
            
            rewards = [r["reward"] for r in history]
            axes[idx].hist(rewards, bins=30, alpha=0.7, edgecolor='black')
            axes[idx].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.3f}')
            axes[idx].set_xlabel("Reward")
            axes[idx].set_ylabel("Frequency")
            axes[idx].set_title(f"{agent_name} - Reward Distribution")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, "reward_distribution.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def plot_action_distribution(self, save_path: str = None):
        """
        Plot action distribution for each agent
        
        Args:
            save_path: Path to save the figure
        """
        if not self.metrics_data:
            return None
        
        action_distribution = self.metrics_data.get("action_distribution", {})
        if not action_distribution:
            return None
        
        # Check if we have any data
        has_data = False
        for agent_name, actions in action_distribution.items():
            if actions and len(actions) > 0:
                has_data = True
                break
        
        if not has_data:
            return None
        
        num_agents = len(action_distribution)
        fig, axes = plt.subplots(1, num_agents, figsize=(6 * num_agents, 5))
        if num_agents == 1:
            axes = [axes]
        
        for idx, (agent_name, actions) in enumerate(action_distribution.items()):
            if not actions:
                continue
            
            action_types = list(actions.keys())
            counts = list(actions.values())
            
            axes[idx].bar(action_types, counts, alpha=0.7, edgecolor='black')
            axes[idx].set_xlabel("Action Type")
            axes[idx].set_ylabel("Count")
            axes[idx].set_title(f"{agent_name} - Action Distribution")
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, "action_distribution.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def generate_summary_table(self, save_path: str = None):
        """
        Generate a summary table as text file
        
        Args:
            save_path: Path to save the table
        """
        if not self.metrics_data:
            return
        
        summary = self.metrics_data.get("summary", {})
        if not summary:
            return
        
        lines = []
        lines.append("=" * 80)
        lines.append("RL Training Summary")
        lines.append("=" * 80)
        lines.append("")
        
        agents = summary.get("agents", {})
        for agent_name, stats in agents.items():
            lines.append(f"Agent: {agent_name}")
            lines.append("-" * 80)
            lines.append(f"  Total Rewards Recorded: {stats.get('total_rewards', 0)}")
            lines.append(f"  Mean Reward: {stats.get('mean_reward', 0.0):.4f}")
            lines.append(f"  Std Reward: {stats.get('std_reward', 0.0):.4f}")
            lines.append(f"  Min Reward: {stats.get('min_reward', 0.0):.4f}")
            lines.append(f"  Max Reward: {stats.get('max_reward', 0.0):.4f}")
            lines.append(f"  Total Episodes: {stats.get('total_episodes', 0)}")
            lines.append(f"  Mean Episode Reward: {stats.get('mean_episode_reward', 0.0):.4f}")
            lines.append("")
            
            # Reward components
            components = stats.get("reward_components", {})
            if components:
                lines.append("  Reward Components:")
                for component, comp_stats in components.items():
                    lines.append(f"    {component}:")
                    lines.append(f"      Mean: {comp_stats.get('mean', 0.0):.4f}")
                    lines.append(f"      Std: {comp_stats.get('std', 0.0):.4f}")
                    lines.append(f"      Count: {comp_stats.get('count', 0)}")
                lines.append("")
            
            # Action distribution
            actions = stats.get("action_distribution", {})
            if actions:
                lines.append("  Action Distribution:")
                for action_type, count in actions.items():
                    lines.append(f"    {action_type}: {count}")
                lines.append("")
        
        lines.append("=" * 80)
        
        content = "\n".join(lines)
        
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            save_path = os.path.join(self.output_dir, "rl_summary.txt")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)
        
        return save_path
    
    def generate_all_visualizations(self):
        """Generate all visualizations and summary"""
        if not self.metrics_data:
            print("No metrics data loaded")
            return
        
        # Check if we have any data
        reward_history = self.metrics_data.get("reward_history", {})
        if not reward_history:
            print("Warning: No reward history data found. Cannot generate visualizations.")
            print("This might mean:")
            print("  1. No rewards were recorded during simulation")
            print("  2. RL collector was not properly initialized")
            print("  3. Metrics recorder was not enabled")
            return
        
        # Check if we have any actual reward data
        has_data = False
        for agent_name, history in reward_history.items():
            if history and len(history) > 0:
                has_data = True
                break
        
        if not has_data:
            print("Warning: Reward history exists but is empty. Cannot generate visualizations.")
            return
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Generating RL metrics visualizations...")
        
        # Generate all plots (with error handling)
        try:
            result = self.plot_reward_trends()
            if result:
                print(f"  ✓ Reward trends plot saved: {result}")
            else:
                print("  ⚠ Reward trends plot skipped (no data)")
        except Exception as e:
            print(f"  ✗ Failed to generate reward trends: {e}")
        
        try:
            result = self.plot_reward_components()
            if result:
                print(f"  ✓ Reward components plot saved: {result}")
            else:
                print("  ⚠ Reward components plot skipped (no data)")
        except Exception as e:
            print(f"  ✗ Failed to generate reward components: {e}")
        
        try:
            result = self.plot_reward_distribution()
            if result:
                print(f"  ✓ Reward distribution plot saved: {result}")
            else:
                print("  ⚠ Reward distribution plot skipped (no data)")
        except Exception as e:
            print(f"  ✗ Failed to generate reward distribution: {e}")
        
        try:
            result = self.plot_action_distribution()
            if result:
                print(f"  ✓ Action distribution plot saved: {result}")
            else:
                print("  ⚠ Action distribution plot skipped (no data)")
        except Exception as e:
            print(f"  ✗ Failed to generate action distribution: {e}")
        
        # Generate summary table
        try:
            result = self.generate_summary_table()
            if result:
                print(f"  ✓ Summary table saved: {result}")
            else:
                print("  ⚠ Summary table skipped (no data)")
        except Exception as e:
            print(f"  ✗ Failed to generate summary table: {e}")
        
        print(f"\nAll visualizations saved to: {self.output_dir}")

