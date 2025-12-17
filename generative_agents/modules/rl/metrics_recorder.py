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
        
        # Reset episode rewards
        self.current_episode_rewards = defaultdict(float)
    
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

