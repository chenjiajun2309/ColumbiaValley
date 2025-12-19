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
    
    def plot_learning_curves(self, save_path: str = None, baseline_metrics_file: str = None):
        """
        Plot learning curves (episode return vs training step)
        
        This is the standard RL paper plot showing how agents learn over time.
        
        Args:
            save_path: Path to save the figure
            baseline_metrics_file: Optional path to baseline (no RL) metrics for comparison
        """
        if not self.metrics_data:
            return None
        
        episode_returns = self.metrics_data.get("episode_returns", {})
        if not episode_returns:
            return None
        
        # Check if we have any data
        has_data = False
        for agent_name, returns in episode_returns.items():
            if returns and len(returns) > 0:
                has_data = True
                break
        
        if not has_data:
            return None
        
        # Load baseline data if provided
        baseline_data = None
        if baseline_metrics_file and os.path.exists(baseline_metrics_file):
            try:
                with open(baseline_metrics_file, "r", encoding="utf-8") as f:
                    baseline_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load baseline metrics: {e}")
        
        fig, axes = plt.subplots(len(episode_returns), 1, figsize=(12, 4 * len(episode_returns)))
        if len(episode_returns) == 1:
            axes = [axes]
        
        for idx, (agent_name, returns) in enumerate(episode_returns.items()):
            if not returns:
                continue
            
            # Extract episode numbers and returns
            episodes = [r["episode"] for r in returns]
            returns_values = [r["return"] for r in returns]
            steps = [r["step"] for r in returns]
            
            # Plot RL learning curve
            axes[idx].plot(episodes, returns_values, alpha=0.7, linewidth=2, 
                          label="RL (with MAPPO)", color='blue')
            
            # Add moving average for RL
            if len(returns_values) > 5:
                window = min(10, len(returns_values) // 5)
                moving_avg = np.convolve(returns_values, np.ones(window)/window, mode='valid')
                moving_episodes = episodes[window-1:]
                axes[idx].plot(moving_episodes, moving_avg, linewidth=2.5, 
                             label="RL Moving Average", color='darkblue', linestyle='--')
            
            # Plot baseline if available
            if baseline_data:
                baseline_returns = baseline_data.get("episode_returns", {}).get(agent_name, [])
                if baseline_returns:
                    baseline_episodes = [r["episode"] for r in baseline_returns]
                    baseline_returns_values = [r["return"] for r in baseline_returns]
                    axes[idx].plot(baseline_episodes, baseline_returns_values, 
                                 alpha=0.7, linewidth=2, label="Baseline (no RL)", 
                                 color='red', linestyle=':')
                    
                    # Add moving average for baseline
                    if len(baseline_returns_values) > 5:
                        window = min(10, len(baseline_returns_values) // 5)
                        baseline_moving_avg = np.convolve(
                            baseline_returns_values, np.ones(window)/window, mode='valid'
                        )
                        baseline_moving_episodes = baseline_episodes[window-1:]
                        axes[idx].plot(baseline_moving_episodes, baseline_moving_avg, 
                                     linewidth=2.5, label="Baseline Moving Average", 
                                     color='darkred', linestyle='--')
            
            axes[idx].set_xlabel("Episode", fontsize=12)
            axes[idx].set_ylabel("Cumulative Return", fontsize=12)
            axes[idx].set_title(f"{agent_name} - Learning Curve", fontsize=14, fontweight='bold')
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, "learning_curves.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def plot_training_losses(self, save_path: str = None):
        """
        Plot training loss curves (policy loss, value loss, entropy)
        
        Args:
            save_path: Path to save the figure
        """
        if not self.metrics_data:
            return None
        
        training_losses = self.metrics_data.get("training_losses", {})
        if not training_losses:
            return None
        
        # Check if we have any data
        has_data = False
        for agent_name, losses in training_losses.items():
            if losses and len(losses) > 0:
                has_data = True
                break
        
        if not has_data:
            return None
        
        fig, axes = plt.subplots(len(training_losses), 1, figsize=(12, 4 * len(training_losses)))
        if len(training_losses) == 1:
            axes = [axes]
        
        for idx, (agent_name, losses) in enumerate(training_losses.items()):
            if not losses:
                continue
            
            steps = [l["step"] for l in losses]
            policy_losses = [l["policy_loss"] for l in losses]
            value_losses = [l["value_loss"] for l in losses]
            entropies = [l.get("entropy") for l in losses if l.get("entropy") is not None]
            entropy_steps = [l["step"] for l in losses if l.get("entropy") is not None]
            
            # Plot policy loss
            axes[idx].plot(steps, policy_losses, alpha=0.7, linewidth=1.5, 
                          label="Policy Loss", color='blue')
            
            # Plot value loss
            axes[idx].plot(steps, value_losses, alpha=0.7, linewidth=1.5, 
                          label="Value Loss", color='red')
            
            # Plot entropy if available
            if entropies:
                ax2 = axes[idx].twinx()
                ax2.plot(entropy_steps, entropies, alpha=0.7, linewidth=1.5, 
                        label="Entropy", color='green')
                ax2.set_ylabel("Entropy", fontsize=10, color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                ax2.legend(loc='upper right', fontsize=9)
            
            axes[idx].set_xlabel("Training Step", fontsize=12)
            axes[idx].set_ylabel("Loss", fontsize=12)
            axes[idx].set_title(f"{agent_name} - Training Losses", fontsize=14, fontweight='bold')
            axes[idx].legend(loc='upper left', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, "training_losses.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def plot_value_estimates(self, save_path: str = None):
        """
        Plot value function estimates over training steps
        
        Args:
            save_path: Path to save the figure
        """
        if not self.metrics_data:
            return None
        
        value_estimates = self.metrics_data.get("value_estimates", {})
        if not value_estimates:
            return None
        
        # Check if we have any data
        has_data = False
        for agent_name, estimates in value_estimates.items():
            if estimates and len(estimates) > 0:
                has_data = True
                break
        
        if not has_data:
            return None
        
        fig, axes = plt.subplots(len(value_estimates), 1, figsize=(12, 4 * len(value_estimates)))
        if len(value_estimates) == 1:
            axes = [axes]
        
        for idx, (agent_name, estimates) in enumerate(value_estimates.items()):
            if not estimates:
                continue
            
            steps = [e["step"] for e in estimates]
            mean_values = [e["mean_value"] for e in estimates]
            std_values = [e["std_value"] for e in estimates]
            
            axes[idx].plot(steps, mean_values, alpha=0.7, linewidth=2, 
                          label="Mean Value", color='blue')
            if any(std > 0 for std in std_values):
                axes[idx].fill_between(steps, 
                                       [m - s for m, s in zip(mean_values, std_values)],
                                       [m + s for m, s in zip(mean_values, std_values)],
                                       alpha=0.3, color='blue', label="±1 Std")
            
            axes[idx].set_xlabel("Training Step", fontsize=12)
            axes[idx].set_ylabel("Value Estimate", fontsize=12)
            axes[idx].set_title(f"{agent_name} - Value Function Estimates", fontsize=14, fontweight='bold')
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, "value_estimates.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def plot_kl_divergence(self, save_path: str = None):
        """
        Plot KL divergence between old and new policies
        
        Args:
            save_path: Path to save the figure
        """
        if not self.metrics_data:
            return None
        
        kl_divergences = self.metrics_data.get("kl_divergences", {})
        if not kl_divergences:
            return None
        
        # Check if we have any data
        has_data = False
        for agent_name, kls in kl_divergences.items():
            if kls and len(kls) > 0:
                has_data = True
                break
        
        if not has_data:
            return None
        
        fig, axes = plt.subplots(len(kl_divergences), 1, figsize=(12, 4 * len(kl_divergences)))
        if len(kl_divergences) == 1:
            axes = [axes]
        
        for idx, (agent_name, kls) in enumerate(kl_divergences.items()):
            if not kls:
                continue
            
            steps = [k["step"] for k in kls]
            kl_values = [k["kl_divergence"] for k in kls]
            
            axes[idx].plot(steps, kl_values, alpha=0.7, linewidth=2, 
                          label="KL Divergence", color='purple')
            axes[idx].axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            axes[idx].set_xlabel("Training Step", fontsize=12)
            axes[idx].set_ylabel("KL Divergence", fontsize=12)
            axes[idx].set_title(f"{agent_name} - Policy KL Divergence", fontsize=14, fontweight='bold')
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, "kl_divergence.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def plot_gradient_norms(self, save_path: str = None):
        """
        Plot gradient norms for policy and value networks
        
        Args:
            save_path: Path to save the figure
        """
        if not self.metrics_data:
            return None
        
        gradient_norms = self.metrics_data.get("gradient_norms", {})
        if not gradient_norms:
            return None
        
        # Check if we have any data
        has_data = False
        for agent_name, norms in gradient_norms.items():
            if norms and len(norms) > 0:
                has_data = True
                break
        
        if not has_data:
            return None
        
        fig, axes = plt.subplots(len(gradient_norms), 1, figsize=(12, 4 * len(gradient_norms)))
        if len(gradient_norms) == 1:
            axes = [axes]
        
        for idx, (agent_name, norms) in enumerate(gradient_norms.items()):
            if not norms:
                continue
            
            steps = [n["step"] for n in norms]
            policy_norms = [n["policy_grad_norm"] for n in norms]
            value_norms = [n["value_grad_norm"] for n in norms]
            
            axes[idx].plot(steps, policy_norms, alpha=0.7, linewidth=2, 
                          label="Policy Gradient Norm", color='blue')
            axes[idx].plot(steps, value_norms, alpha=0.7, linewidth=2, 
                          label="Value Gradient Norm", color='red')
            
            axes[idx].set_xlabel("Training Step", fontsize=12)
            axes[idx].set_ylabel("Gradient Norm", fontsize=12)
            axes[idx].set_title(f"{agent_name} - Gradient Norms", fontsize=14, fontweight='bold')
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_yscale('log')  # Log scale for better visualization
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, "gradient_norms.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def plot_policy_ratios(self, save_path: str = None):
        """
        Plot policy ratios (exp(log_prob_new - log_prob_old)) and clip fraction
        
        Args:
            save_path: Path to save the figure
        """
        if not self.metrics_data:
            return None
        
        policy_ratios = self.metrics_data.get("policy_ratios", {})
        if not policy_ratios:
            return None
        
        # Check if we have any data
        has_data = False
        for agent_name, ratios in policy_ratios.items():
            if ratios and len(ratios) > 0:
                has_data = True
                break
        
        if not has_data:
            return None
        
        fig, axes = plt.subplots(len(policy_ratios), 1, figsize=(12, 4 * len(policy_ratios)))
        if len(policy_ratios) == 1:
            axes = [axes]
        
        for idx, (agent_name, ratios) in enumerate(policy_ratios.items()):
            if not ratios:
                continue
            
            steps = [r["step"] for r in ratios]
            mean_ratios = [r["mean_ratio"] for r in ratios]
            std_ratios = [r["std_ratio"] for r in ratios]
            clip_fractions = [r["clip_fraction"] for r in ratios]
            
            # Plot mean ratio with std
            axes[idx].plot(steps, mean_ratios, alpha=0.7, linewidth=2, 
                          label="Mean Ratio", color='blue')
            if any(std > 0 for std in std_ratios):
                axes[idx].fill_between(steps, 
                                      [m - s for m, s in zip(mean_ratios, std_ratios)],
                                      [m + s for m, s in zip(mean_ratios, std_ratios)],
                                      alpha=0.3, color='blue', label="±1 Std")
            
            # Add reference lines for clipping bounds (typically 0.8 and 1.2 for epsilon=0.2)
            axes[idx].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1, label="Ratio = 1.0")
            axes[idx].axhline(y=0.8, color='red', linestyle=':', alpha=0.5, linewidth=1, label="Clip Bound (0.8)")
            axes[idx].axhline(y=1.2, color='red', linestyle=':', alpha=0.5, linewidth=1, label="Clip Bound (1.2)")
            
            # Plot clip fraction on secondary axis
            ax2 = axes[idx].twinx()
            ax2.plot(steps, clip_fractions, alpha=0.7, linewidth=2, 
                    label="Clip Fraction", color='orange', linestyle='--')
            ax2.set_ylabel("Clip Fraction", fontsize=10, color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax2.set_ylim([0, 1])
            
            axes[idx].set_xlabel("Training Step", fontsize=12)
            axes[idx].set_ylabel("Policy Ratio", fontsize=12)
            axes[idx].set_title(f"{agent_name} - Policy Ratios & Clip Fraction", fontsize=14, fontweight='bold')
            axes[idx].legend(loc='upper left', fontsize=9)
            ax2.legend(loc='upper right', fontsize=9)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, "policy_ratios.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def generate_all_visualizations(self, baseline_metrics_file: str = None):
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
        
        try:
            result = self.plot_learning_curves(baseline_metrics_file=baseline_metrics_file)
            if result:
                print(f"  ✓ Learning curves plot saved: {result}")
            else:
                print("  ⚠ Learning curves plot skipped (no data)")
        except Exception as e:
            print(f"  ✗ Failed to generate learning curves: {e}")
        
        try:
            result = self.plot_training_losses()
            if result:
                print(f"  ✓ Training loss curves plot saved: {result}")
            else:
                print("  ⚠ Training loss curves plot skipped (no data)")
        except Exception as e:
            print(f"  ✗ Failed to generate training loss curves: {e}")
        
        # Advanced visualizations (if data available)
        try:
            result = self.plot_value_estimates()
            if result:
                print(f"  ✓ Value estimates plot saved: {result}")
        except Exception as e:
            print(f"  ✗ Failed to generate value estimates: {e}")
        
        try:
            result = self.plot_kl_divergence()
            if result:
                print(f"  ✓ KL divergence plot saved: {result}")
        except Exception as e:
            print(f"  ✗ Failed to generate KL divergence: {e}")
        
        try:
            result = self.plot_gradient_norms()
            if result:
                print(f"  ✓ Gradient norms plot saved: {result}")
        except Exception as e:
            print(f"  ✗ Failed to generate gradient norms: {e}")
        
        try:
            result = self.plot_policy_ratios()
            if result:
                print(f"  ✓ Policy ratios plot saved: {result}")
        except Exception as e:
            print(f"  ✗ Failed to generate policy ratios: {e}")
        
        print(f"\nAll visualizations saved to: {self.output_dir}")

