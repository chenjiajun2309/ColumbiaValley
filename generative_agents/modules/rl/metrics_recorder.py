"""
RL Metrics Recorder

Records and tracks RL training metrics for visualization and analysis.
"""

import json
import os
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np


class RLMetricsRecorder:
    """
    Record RL training metrics for visualization
    
    Tracks:
    - Reward history per agent
    - Reward components (persona, interaction, relationship, etc.)
    - Training statistics
    - Agent behavior metrics
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.checkpoints_folder = self.config.get("checkpoints_folder", "")
        
        # Reward history: {agent_name: [(step, reward, components_dict), ...]}
        self.reward_history = defaultdict(list)
        
        # Training statistics: {step: {agent_name: stats_dict}}
        self.training_stats = {}
        
        # Episode rewards: {agent_name: [episode_rewards]}
        self.episode_rewards = defaultdict(list)
        
        # Current episode rewards
        self.current_episode_rewards = defaultdict(float)
        
        # Reward components history: {agent_name: {component: [values]}}
        self.reward_components = defaultdict(lambda: defaultdict(list))
        
        # Action distribution: {agent_name: {action_type: count}}
        self.action_distribution = defaultdict(lambda: defaultdict(int))
        
        # Training step counter
        self.training_step = 0
        
        # Episode returns: {agent_name: [(episode, return, step), ...]}
        # Episode is defined as a sequence of steps between training intervals
        self.episode_returns = defaultdict(list)
        
        # Current episode return accumulator
        self.current_episode_returns = defaultdict(float)
        self.current_episode_steps = defaultdict(int)
        
        # Training losses: {agent_name: [(step, policy_loss, value_loss, entropy), ...]}
        self.training_losses = defaultdict(list)
        
        # Additional training metrics for advanced visualizations
        # Value estimates: {agent_name: [(step, mean_value, std_value), ...]}
        self.value_estimates = defaultdict(list)
        
        # Advantage estimates: {agent_name: [(step, mean_advantage, std_advantage), ...]}
        self.advantage_estimates = defaultdict(list)
        
        # KL divergence: {agent_name: [(step, kl_divergence), ...]}
        self.kl_divergences = defaultdict(list)
        
        # Gradient norms: {agent_name: [(step, policy_grad_norm, value_grad_norm), ...]}
        self.gradient_norms = defaultdict(list)
        
        # Policy ratios: {agent_name: [(step, mean_ratio, std_ratio, clip_fraction), ...]}
        self.policy_ratios = defaultdict(list)
        
        # Enable/disable recording
        self.enabled = True
    
    def record_reward(
        self,
        agent_name: str,
        reward: float,
        components: Dict[str, float] = None,
        step: int = None
    ):
        """
        Record a reward for an agent
        
        Args:
            agent_name: Name of the agent
            reward: Total reward value
            components: Dictionary of reward components (optional)
            step: Simulation step (optional, uses current training step if not provided)
        """
        if not self.enabled:
            return
        
        if step is None:
            step = self.training_step
        
        # Record reward
        self.reward_history[agent_name].append({
            "step": step,
            "reward": reward,
            "timestamp": time.time(),
            "components": components or {}
        })
        
        # Record components
        if components:
            for component, value in components.items():
                self.reward_components[agent_name][component].append({
                    "step": step,
                    "value": value,
                    "timestamp": time.time()
                })
        
        # Update current episode reward
        self.current_episode_rewards[agent_name] += reward
        
        # Update current episode return (cumulative reward)
        self.current_episode_returns[agent_name] += reward
        self.current_episode_steps[agent_name] += 1
    
    def record_action(self, agent_name: str, action_type: str):
        """
        Record an action taken by an agent
        
        Args:
            agent_name: Name of the agent
            action_type: Type of action taken
        """
        if not self.enabled:
            return
        
        self.action_distribution[agent_name][action_type] += 1
    
    def record_training_step(
        self,
        step: int,
        agent_rewards: Dict[str, float],
        agent_components: Dict[str, Dict[str, float]] = None
    ):
        """
        Record metrics for a training step
        
        Args:
            step: Training step number
            agent_rewards: Dictionary of {agent_name: total_reward}
            agent_components: Dictionary of {agent_name: {component: value}}
        """
        if not self.enabled:
            return
        
        self.training_step = step
        
        stats = {}
        for agent_name, reward in agent_rewards.items():
            # Calculate statistics
            agent_history = self.reward_history[agent_name]
            if agent_history:
                recent_rewards = [r["reward"] for r in agent_history[-10:]]  # Last 10 rewards
                stats[agent_name] = {
                    "current_reward": reward,
                    "mean_reward": np.mean(recent_rewards) if recent_rewards else 0.0,
                    "std_reward": np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0,
                    "min_reward": np.min(recent_rewards) if recent_rewards else 0.0,
                    "max_reward": np.max(recent_rewards) if recent_rewards else 0.0,
                    "total_rewards": len(agent_history),
                    "components": agent_components.get(agent_name, {}) if agent_components else {}
                }
            else:
                stats[agent_name] = {
                    "current_reward": reward,
                    "mean_reward": 0.0,
                    "std_reward": 0.0,
                    "min_reward": 0.0,
                    "max_reward": 0.0,
                    "total_rewards": 0,
                    "components": agent_components.get(agent_name, {}) if agent_components else {}
                }
        
        self.training_stats[step] = stats
        
        # Update episode rewards
        for agent_name, reward in agent_rewards.items():
            self.current_episode_rewards[agent_name] += reward
    
    def end_episode(self, step: int):
        """
        End current episode and record episode rewards
        
        Args:
            step: Current simulation step
        """
        if not self.enabled:
            return
        
        for agent_name, episode_reward in self.current_episode_rewards.items():
            self.episode_rewards[agent_name].append({
                "step": step,
                "reward": episode_reward
            })
        
        # Record episode return (cumulative reward)
        for agent_name, episode_return in self.current_episode_returns.items():
            if episode_return != 0 or self.current_episode_steps[agent_name] > 0:
                episode_num = len(self.episode_returns[agent_name])
                self.episode_returns[agent_name].append({
                    "episode": episode_num,
                    "return": episode_return,
                    "step": step,
                    "steps_in_episode": self.current_episode_steps[agent_name]
                })
        
        # Reset episode rewards and returns
        self.current_episode_rewards = defaultdict(float)
        self.current_episode_returns = defaultdict(float)
        self.current_episode_steps = defaultdict(int)
    
    def record_training_loss(
        self,
        agent_name: str,
        step: int,
        policy_loss: float,
        value_loss: float,
        entropy: float = None
    ):
        """
        Record training loss for an agent
        
        Args:
            agent_name: Name of the agent
            step: Training step number
            policy_loss: Policy loss value
            value_loss: Value loss value
            entropy: Policy entropy (optional)
        """
        if not self.enabled:
            return
        
        self.training_losses[agent_name].append({
            "step": step,
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy": float(entropy) if entropy is not None else None,
            "timestamp": time.time()
        })
    
    def record_training_metrics(
        self,
        agent_name: str,
        step: int,
        mean_value: float = None,
        std_value: float = None,
        mean_advantage: float = None,
        std_advantage: float = None,
        kl_divergence: float = None,
        policy_grad_norm: float = None,
        value_grad_norm: float = None,
        mean_ratio: float = None,
        std_ratio: float = None,
        clip_fraction: float = None
    ):
        """
        Record additional training metrics for advanced visualizations
        
        Args:
            agent_name: Name of the agent
            step: Training step number
            mean_value: Mean value estimate
            std_value: Std of value estimates
            mean_advantage: Mean advantage estimate
            std_advantage: Std of advantage estimates
            kl_divergence: KL divergence between old and new policy
            policy_grad_norm: Policy gradient norm
            value_grad_norm: Value gradient norm
            mean_ratio: Mean policy ratio (exp(log_prob_new - log_prob_old))
            std_ratio: Std of policy ratios
            clip_fraction: Fraction of ratios that were clipped
        """
        if not self.enabled:
            return
        
        if mean_value is not None or std_value is not None:
            self.value_estimates[agent_name].append({
                "step": step,
                "mean_value": float(mean_value) if mean_value is not None else 0.0,
                "std_value": float(std_value) if std_value is not None else 0.0,
                "timestamp": time.time()
            })
        
        if mean_advantage is not None or std_advantage is not None:
            self.advantage_estimates[agent_name].append({
                "step": step,
                "mean_advantage": float(mean_advantage) if mean_advantage is not None else 0.0,
                "std_advantage": float(std_advantage) if std_advantage is not None else 0.0,
                "timestamp": time.time()
            })
        
        if kl_divergence is not None:
            self.kl_divergences[agent_name].append({
                "step": step,
                "kl_divergence": float(kl_divergence),
                "timestamp": time.time()
            })
        
        if policy_grad_norm is not None or value_grad_norm is not None:
            self.gradient_norms[agent_name].append({
                "step": step,
                "policy_grad_norm": float(policy_grad_norm) if policy_grad_norm is not None else 0.0,
                "value_grad_norm": float(value_grad_norm) if value_grad_norm is not None else 0.0,
                "timestamp": time.time()
            })
        
        if mean_ratio is not None or std_ratio is not None or clip_fraction is not None:
            self.policy_ratios[agent_name].append({
                "step": step,
                "mean_ratio": float(mean_ratio) if mean_ratio is not None else 1.0,
                "std_ratio": float(std_ratio) if std_ratio is not None else 0.0,
                "clip_fraction": float(clip_fraction) if clip_fraction is not None else 0.0,
                "timestamp": time.time()
            })
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "total_training_steps": self.training_step,
            "agents": {}
        }
        
        for agent_name in self.reward_history.keys():
            rewards = [r["reward"] for r in self.reward_history[agent_name]]
            if rewards:
                summary["agents"][agent_name] = {
                    "total_rewards": len(rewards),
                    "mean_reward": float(np.mean(rewards)),
                    "std_reward": float(np.std(rewards)),
                    "min_reward": float(np.min(rewards)),
                    "max_reward": float(np.max(rewards)),
                    "total_episodes": len(self.episode_rewards[agent_name]),
                    "mean_episode_reward": float(np.mean([e["reward"] for e in self.episode_rewards[agent_name]])) if self.episode_rewards[agent_name] else 0.0,
                    "action_distribution": dict(self.action_distribution[agent_name]),
                    "reward_components": {
                        component: {
                            "mean": float(np.mean([v["value"] for v in values])) if values else 0.0,
                            "std": float(np.std([v["value"] for v in values])) if len(values) > 1 else 0.0,
                            "count": len(values)
                        }
                        for component, values in self.reward_components[agent_name].items()
                    }
                }
        
        return summary
    
    def save_metrics(self, filepath: str = None):
        """
        Save metrics to JSON file
        
        Args:
            filepath: Path to save metrics (optional)
        """
        # End the final episode before saving (for both RL and baseline)
        # This ensures the last episode is recorded even if it doesn't align with intervals
        if self.current_episode_returns and any(v > 0 or self.current_episode_steps.get(k, 0) > 0 
                                                for k, v in self.current_episode_returns.items()):
            final_step = self.training_step if hasattr(self, 'training_step') else 0
            self.end_episode(final_step)
        
        if not filepath:
            if not self.checkpoints_folder:
                print(f"Warning: checkpoints_folder is empty, cannot save metrics")
                return None
            filepath = os.path.join(self.checkpoints_folder, "rl_metrics.json")
        
        # Ensure directory exists
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        data = {
            "reward_history": {
                agent: history for agent, history in self.reward_history.items()
            },
            "training_stats": self.training_stats,
            "episode_rewards": {
                agent: episodes for agent, episodes in self.episode_rewards.items()
            },
            "episode_returns": {
                agent: returns for agent, returns in self.episode_returns.items()
            },
            "training_losses": {
                agent: losses for agent, losses in self.training_losses.items()
            },
            "value_estimates": {
                agent: estimates for agent, estimates in self.value_estimates.items()
            },
            "advantage_estimates": {
                agent: estimates for agent, estimates in self.advantage_estimates.items()
            },
            "kl_divergences": {
                agent: kls for agent, kls in self.kl_divergences.items()
            },
            "gradient_norms": {
                agent: norms for agent, norms in self.gradient_norms.items()
            },
            "policy_ratios": {
                agent: ratios for agent, ratios in self.policy_ratios.items()
            },
            "reward_components": {
                agent: {
                    component: values
                    for component, values in components.items()
                }
                for agent, components in self.reward_components.items()
            },
            "action_distribution": {
                agent: dict(actions)
                for agent, actions in self.action_distribution.items()
            },
            "summary": self.get_summary()
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_metrics(self, filepath: str):
        """
        Load metrics from JSON file
        
        Args:
            filepath: Path to load metrics from
        """
        if not os.path.exists(filepath):
            return
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.reward_history = defaultdict(list, data.get("reward_history", {}))
        self.training_stats = data.get("training_stats", {})
        self.episode_rewards = defaultdict(list, data.get("episode_rewards", {}))
        self.episode_returns = defaultdict(list, data.get("episode_returns", {}))
        self.training_losses = defaultdict(list, data.get("training_losses", {}))
        self.value_estimates = defaultdict(list, data.get("value_estimates", {}))
        self.advantage_estimates = defaultdict(list, data.get("advantage_estimates", {}))
        self.kl_divergences = defaultdict(list, data.get("kl_divergences", {}))
        self.gradient_norms = defaultdict(list, data.get("gradient_norms", {}))
        self.policy_ratios = defaultdict(list, data.get("policy_ratios", {}))
        self.reward_components = defaultdict(
            lambda: defaultdict(list),
            {
                agent: {
                    component: values
                    for component, values in components.items()
                }
                for agent, components in data.get("reward_components", {}).items()
            }
        )
        self.action_distribution = defaultdict(
            lambda: defaultdict(int),
            {
                agent: dict(actions)
                for agent, actions in data.get("action_distribution", {}).items()
            }
        )

