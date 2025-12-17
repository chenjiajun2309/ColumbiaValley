"""
MAPPO Trainer - Simplified implementation example

Note: This is a conceptual implementation. For production use,
consider using established libraries like Stable-Baselines3 or RLLib.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from modules.agent import Agent
from modules.game import Game
from modules.rl.state_extractor import StateExtractor
from modules.rl.action_space import ActionSpace
from modules.rl.reward_function import RewardFunction
from modules.rl.mappo.network import PolicyNetwork, ValueNetwork


class MAPPOTrainer:
    """
    MAPPO Trainer for multi-agent reinforcement learning
    
    This is a simplified implementation. For production, you should:
    1. Use established RL libraries (Stable-Baselines3, RLLib)
    2. Implement proper GAE (Generalized Advantage Estimation)
    3. Add proper normalization and clipping
    4. Implement experience replay properly
    """
    
    def __init__(
        self,
        agents: Dict[str, Agent],
        game: Game,
        config: Dict = None
    ):
        self.agents = agents
        self.game = game
        self.config = config or {}
        
        # Device configuration (GPU if available, else CPU)
        self.device = self._get_device(config.get("device", None))
        print(f"🔧 Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        
        # RL components
        self.state_extractor = StateExtractor(config.get("state_extractor", {}))
        self.action_space = ActionSpace(config.get("action_space", {}))
        self.reward_function = RewardFunction(config.get("reward_function", {}))
        
        # Networks (one per agent, or shared)
        self.use_shared_network = config.get("use_shared_network", True)
        if self.use_shared_network:
            # Shared policy and value networks
            state_dim = self._get_state_dim()
            action_dim = self.action_space.get_action_space_size()
            self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
            self.value_net = ValueNetwork(state_dim).to(self.device)
            self.policies = {name: self.policy_net for name in agents}
            self.values = {name: self.value_net for name in agents}
        else:
            # Separate networks per agent
            state_dim = self._get_state_dim()
            action_dim = self.action_space.get_action_space_size()
            self.policies = {
                name: PolicyNetwork(state_dim, action_dim).to(self.device)
                for name in agents
            }
            self.values = {
                name: ValueNetwork(state_dim).to(self.device)
                for name in agents
            }
        
        # Optimizers
        self.policy_optimizers = {
            name: torch.optim.Adam(policy.parameters(), lr=config.get("lr", 3e-4))
            for name, policy in self.policies.items()
        }
        self.value_optimizers = {
            name: torch.optim.Adam(value.parameters(), lr=config.get("lr", 3e-4))
            for name, value in self.values.items()
        }
    
    def _get_device(self, device_config: str = None) -> torch.device:
        """
        Get the device to use for training
        
        Args:
            device_config: Device configuration from config ("cuda", "cpu", or None for auto)
            
        Returns:
            torch.device object
        """
        if device_config is not None:
            if device_config.lower() == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            elif device_config.lower() == "cpu":
                return torch.device("cpu")
            elif device_config.startswith("cuda:"):
                # Specific GPU device
                device_id = int(device_config.split(":")[1])
                if device_id < torch.cuda.device_count():
                    return torch.device(f"cuda:{device_id}")
        
        # Auto-detect: use GPU if available, else CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
        
        # Training buffers
        self.buffers = {name: [] for name in agents}
        
        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.num_epochs = config.get("num_epochs", 10)
        self.batch_size = config.get("batch_size", 64)
        self.rollout_length = config.get("rollout_length", 2048)
    
    def _get_state_dim(self) -> int:
        """Get state dimension (simplified)"""
        # This should match StateExtractor.state_to_vector output
        # For now, return a placeholder
        return 20  # Adjust based on actual state features
    
    def collect_rollout(self, num_steps: int) -> Dict[str, List]:
        """
        Collect rollout data from environment
        
        Args:
            num_steps: Number of steps to collect
            
        Returns:
            Dictionary of rollout data per agent
        """
        rollouts = {name: [] for name in self.agents}
        
        for step in range(num_steps):
            # Extract states for all agents
            states = {}
            for name, agent in self.agents.items():
                state = self.state_extractor.extract(agent, self.game)
                states[name] = state
            
            # Get actions from policies
            actions = {}
            action_log_probs = {}
            values = {}
            
            for name, agent in self.agents.items():
                state_vec = self.state_extractor.state_to_vector(states[name])
                state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
                
                # Get action from policy
                action_dist = self.policies[name](state_tensor)
                action = action_dist.sample()
                action_log_prob = action_dist.log_prob(action)
                value = self.values[name](state_tensor)
                
                actions[name] = action.item()
                action_log_probs[name] = action_log_prob.item()
                values[name] = value.item()
            
            # Execute actions (simplified - in practice, integrate with Agent.think)
            # This is where you'd modify agent behavior based on RL actions
            next_states = {}
            rewards = {}
            dones = {}
            
            for name, agent in self.agents.items():
                # Decode action
                action_dict = self.action_space.decode_action(
                    actions[name], agent, self.game
                )
                
                # Apply action (this would modify agent state)
                # In practice, you'd integrate this with Agent.think()
                # For now, we'll just simulate
                previous_state = states[name]
                
                # Simulate next state (in practice, this comes from game step)
                # You'd call game.agent_think() here
                next_state = self.state_extractor.extract(agent, self.game)
                next_states[name] = next_state
                
                # Compute reward
                reward = self.reward_function.compute_reward(
                    agent, action_dict, next_state, self.game, previous_state
                )
                rewards[name] = reward
                dones[name] = False  # In practice, check if episode is done
            
            # Store in buffers
            for name in self.agents:
                rollouts[name].append({
                    "state": states[name],
                    "action": actions[name],
                    "action_log_prob": action_log_probs[name],
                    "value": values[name],
                    "reward": rewards[name],
                    "done": dones[name],
                    "next_state": next_states[name],
                })
        
        return rollouts
    
    def compute_advantages(self, rollouts: Dict[str, List]) -> Dict[str, List]:
        """
        Compute advantages using GAE (Generalized Advantage Estimation)
        
        Args:
            rollouts: Rollout data per agent
            
        Returns:
            Advantages and returns per agent
        """
        advantages = {name: [] for name in self.agents}
        returns = {name: [] for name in self.agents}
        
        for name in self.agents:
            agent_rollout = rollouts[name]
            
            # Compute returns (discounted rewards)
            G = 0
            for step in reversed(agent_rollout):
                G = step["reward"] + self.gamma * G * (1 - step["done"])
                returns[name].insert(0, G)
            
            # Compute advantages using GAE
            gae = 0
            for i in reversed(range(len(agent_rollout))):
                step = agent_rollout[i]
                delta = step["reward"] + self.gamma * (
                    agent_rollout[i+1]["value"] if i+1 < len(agent_rollout) else 0
                ) * (1 - step["done"]) - step["value"]
                gae = delta + self.gamma * self.gae_lambda * gae * (1 - step["done"])
                advantages[name].insert(0, gae)
        
        return advantages, returns
    
    def train_step(self, rollouts: Dict[str, List]):
        """
        Perform one training step using MAPPO
        
        Args:
            rollouts: Rollout data per agent
        """
        # Compute advantages
        advantages, returns = self.compute_advantages(rollouts)
        
        # Normalize advantages
        for name in self.agents:
            adv = np.array(advantages[name])
            advantages[name] = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        # Train each agent
        for name, agent in self.agents.items():
            agent_rollout = rollouts[name]
            agent_advantages = advantages[name]
            agent_returns = returns[name]
            
            # Convert to tensors and move to device
            states = torch.FloatTensor([
                self.state_extractor.state_to_vector(step["state"])
                for step in agent_rollout
            ]).to(self.device)
            actions = torch.LongTensor([step["action"] for step in agent_rollout]).to(self.device)
            old_log_probs = torch.FloatTensor([step["action_log_prob"] for step in agent_rollout]).to(self.device)
            advantages_tensor = torch.FloatTensor(agent_advantages).to(self.device)
            returns_tensor = torch.FloatTensor(agent_returns).to(self.device)
            
            # Train for multiple epochs
            for epoch in range(self.num_epochs):
                # Shuffle data
                indices = torch.randperm(len(agent_rollout))
                
                for start in range(0, len(agent_rollout), self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages_tensor[batch_indices]
                    batch_returns = returns_tensor[batch_indices]
                    
                    # Get current policy predictions
                    action_dist = self.policies[name](batch_states)
                    new_log_probs = action_dist.log_prob(batch_actions)
                    entropy = action_dist.entropy().mean()
                    
                    # Compute value predictions
                    values = self.values[name](batch_states).squeeze()
                    
                    # Compute policy loss (PPO clip)
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Compute value loss
                    value_loss = nn.MSELoss()(values, batch_returns)
                    
                    # Total loss
                    loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                    
                    # Update policy
                    self.policy_optimizers[name].zero_grad()
                    self.value_optimizers[name].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.policies[name].parameters()) + list(self.values[name].parameters()),
                        self.max_grad_norm
                    )
                    self.policy_optimizers[name].step()
                    self.value_optimizers[name].step()
    
    def save_models(self, path: str):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save device info
        device_info = {"device": str(self.device)}
        import json
        with open(os.path.join(path, "device_info.json"), "w") as f:
            json.dump(device_info, f)
        
        if self.use_shared_network:
            torch.save(self.policy_net.state_dict(), os.path.join(path, "policy_net.pt"))
            torch.save(self.value_net.state_dict(), os.path.join(path, "value_net.pt"))
        else:
            for name in self.agents:
                torch.save(
                    self.policies[name].state_dict(),
                    os.path.join(path, f"{name}_policy.pt")
                )
                torch.save(
                    self.values[name].state_dict(),
                    os.path.join(path, f"{name}_value.pt")
                )
    
    def load_models(self, path: str):
        """Load trained models"""
        import os
        
        # Load device info if available
        device_info_path = os.path.join(path, "device_info.json")
        if os.path.exists(device_info_path):
            import json
            with open(device_info_path, "r") as f:
                device_info = json.load(f)
                # Note: We use the current device, not the saved device
        
        if self.use_shared_network:
            self.policy_net.load_state_dict(
                torch.load(os.path.join(path, "policy_net.pt"), map_location=self.device)
            )
            self.value_net.load_state_dict(
                torch.load(os.path.join(path, "value_net.pt"), map_location=self.device)
            )
        else:
            for name in self.agents:
                self.policies[name].load_state_dict(
                    torch.load(os.path.join(path, f"{name}_policy.pt"), map_location=self.device)
                )
                self.values[name].load_state_dict(
                    torch.load(os.path.join(path, f"{name}_value.pt"), map_location=self.device)
                )

