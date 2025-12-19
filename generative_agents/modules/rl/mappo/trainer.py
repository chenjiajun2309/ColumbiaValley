"""MAPPO Trainer Implementation"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .network import PolicyNetwork, ValueNetwork


class MAPPOTrainer:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) Trainer
    
    Implements MAPPO algorithm for training multiple agents with independent policies.
    """
    
    def __init__(self, config: Dict = None, metrics_recorder=None):
        """
        Initialize MAPPO trainer
        
        Args:
            config: Configuration dictionary
            metrics_recorder: Optional metrics recorder for logging training losses
        """
        self.config = config or {}
        
        # Training hyperparameters
        self.gamma = self.config.get("gamma", 0.99)  # Discount factor
        self.gae_lambda = self.config.get("gae_lambda", 0.95)  # GAE lambda
        self.clip_epsilon = self.config.get("clip_epsilon", 0.2)  # PPO clip epsilon
        self.value_loss_coef = self.config.get("value_loss_coef", 0.5)  # Value loss coefficient
        self.entropy_coef = self.config.get("entropy_coef", 0.01)  # Entropy coefficient
        self.max_grad_norm = self.config.get("max_grad_norm", 0.5)  # Gradient clipping
        self.policy_lr = self.config.get("policy_lr", 3e-4)  # Policy learning rate
        self.value_lr = self.config.get("value_lr", 3e-4)  # Value learning rate
        self.num_epochs = self.config.get("num_epochs", 10)  # Training epochs per update
        self.batch_size = self.config.get("batch_size", 64)  # Batch size
        
        # State and action dimensions (will be set when first training)
        self.state_dim = self.config.get("state_dim", None)
        self.action_dim = self.config.get("action_dim", 6)  # 6 action types
        
        # Agent policies and value networks
        self.policies: Dict[str, PolicyNetwork] = {}
        self.values: Dict[str, ValueNetwork] = {}
        self.policy_optimizers: Dict[str, optim.Adam] = {}
        self.value_optimizers: Dict[str, optim.Adam] = {}
        
        # Old policies for PPO (to compute importance sampling ratio)
        self.old_policies: Dict[str, PolicyNetwork] = {}
        
        # Metrics recorder
        self.metrics_recorder = metrics_recorder
        self.training_step = 0
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MAPPO Trainer initialized on device: {self.device}")
    
    def _initialize_agent_networks(self, agent_name: str, state_dim: int):
        """
        Initialize policy and value networks for an agent
        
        Args:
            agent_name: Name of the agent
            state_dim: State dimension
        """
        if agent_name in self.policies:
            return  # Already initialized
        
        # Initialize networks
        self.policies[agent_name] = PolicyNetwork(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.get("hidden_dim", 64)
        ).to(self.device)
        
        self.values[agent_name] = ValueNetwork(
            state_dim=state_dim,
            hidden_dim=self.config.get("hidden_dim", 64)
        ).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizers[agent_name] = optim.Adam(
            self.policies[agent_name].parameters(),
            lr=self.policy_lr
        )
        
        self.value_optimizers[agent_name] = optim.Adam(
            self.values[agent_name].parameters(),
            lr=self.value_lr
        )
        
        # Initialize old policy (copy of current policy)
        self.old_policies[agent_name] = PolicyNetwork(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.get("hidden_dim", 64)
        ).to(self.device)
        self.old_policies[agent_name].load_state_dict(
            self.policies[agent_name].state_dict()
        )
        self.old_policies[agent_name].eval()
    
    def _extract_state_vector(self, state: Dict) -> np.ndarray:
        """
        Extract state vector from state dictionary
        
        Args:
            state: State dictionary from StateExtractor
                Structure: {
                    "persona_features": {...},
                    "spatial_features": {...},
                    "action_features": {...},
                    "memory_features": {...},
                    "social_features": {...},
                    "schedule_features": {...}
                }
            
        Returns:
            State vector as numpy array
        """
        # Flatten all state features into a single vector
        features = []
        
        # Persona features
        persona = state.get("persona_features", {})
        # Extract persona keywords from currently or persona_summary
        currently = persona.get("currently", "")
        persona_summary = persona.get("persona_summary", "")
        persona_text = (currently + " " + persona_summary).lower()
        features.append(1.0 if "social" in persona_text or "outgoing" in persona_text else 0.0)
        features.append(1.0 if "curious" in persona_text or "explor" in persona_text else 0.0)
        features.append(1.0 if "disciplined" in persona_text or "organized" in persona_text else 0.0)
        features.append(float(len(currently)) / 100.0)  # Normalized currently length
        
        # Spatial features
        spatial = state.get("spatial_features", {})
        coord = spatial.get("coord", [0, 0])
        features.append(float(coord[0]) / 100.0 if len(coord) > 0 else 0.0)  # Normalized x
        features.append(float(coord[1]) / 100.0 if len(coord) > 1 else 0.0)  # Normalized y
        features.append(1.0 if spatial.get("has_path", False) else 0.0)
        
        # Action features
        action = state.get("action_features", {})
        action_type = action.get("action_type", "idle")
        # Map action type to ID (simple mapping)
        action_type_map = {
            "idle": 0, "chat": 1, "wait": 2, "move": 3, "revise": 4, "skip": 5
        }
        action_id = action_type_map.get(action_type.lower(), 0)
        features.append(float(action_id) / 5.0)  # Normalized action ID
        features.append(float(action.get("action_duration", 0)) / 100.0)  # Normalized duration
        features.append(float(action.get("action_progress", 0.0)))  # Progress (0-1)
        
        # Memory features
        memory = state.get("memory_features", {})
        features.append(float(memory.get("num_concepts", 0)) / 50.0)  # Normalized concept count
        features.append(float(memory.get("poignancy", 0)) / 10.0)  # Normalized poignancy
        features.append(float(memory.get("memory_size", 0)) / 1000.0)  # Normalized memory size
        
        # Social features
        social = state.get("social_features", {})
        nearby_agents = social.get("nearby_agents", [])
        relationships = social.get("relationships", {})
        features.append(float(len(nearby_agents)) / 5.0)  # Normalized nearby count
        # Average relationship score
        if relationships:
            avg_rel = sum(r.get("relationship_score", 0) for r in relationships.values()) / len(relationships)
            features.append(float(avg_rel) / 10.0)  # Normalized relationship
        else:
            features.append(0.0)
        
        # Schedule features
        schedule = state.get("schedule_features", {})
        features.append(1.0 if schedule.get("has_schedule", False) else 0.0)
        features.append(float(schedule.get("schedule_progress", 0.0)))  # Progress (0-1)
        features.append(float(schedule.get("daily_progress", 0.0)))  # Daily progress (0-1)
        
        # Pad or truncate to fixed size if needed
        target_dim = self.state_dim or 20  # Default state dimension (adjust based on actual features)
        if len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
        elif len(features) > target_dim:
            features = features[:target_dim]
        
        return np.array(features, dtype=np.float32)
    
    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_values: List[float],
        dones: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            next_values: List of next state value estimates
            dones: List of done flags
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)
        
        # Compute returns (backwards)
        g = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                g = 0
            g = rewards[t] + self.gamma * g
            returns[t] = g
        
        # Compute advantages using GAE (backwards)
        gae = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                gae = 0
            delta = rewards[t] + self.gamma * next_values[t] - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def train_step(self, rollouts: Dict[str, List], step: int = None):
        """
        Train MAPPO on collected rollouts
        
        Args:
            rollouts: Dictionary of {agent_name: [transitions]}
            step: Current training step
        """
        if step is not None:
            self.training_step = step
        
        if not rollouts:
            print("Warning: No rollouts provided for training")
            return
        
        # Process each agent's rollouts
        for agent_name, transitions in rollouts.items():
            if not transitions:
                continue
            
            # Extract data from transitions
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            old_log_probs = []
            values = []
            
            for trans in transitions:
                # Extract state vector
                state_vec = self._extract_state_vector(trans["state"])
                states.append(state_vec)
                
                # Extract action
                actions.append(trans["action"])
                
                # Extract reward
                rewards.append(trans["reward"])
                
                # Extract next state
                next_state_vec = self._extract_state_vector(trans["next_state"])
                next_states.append(next_state_vec)
                
                # Extract done
                dones.append(trans.get("done", False))
                
                # Extract old log prob (if available)
                old_log_probs.append(trans.get("action_log_prob", 0.0))
                
                # Extract value (if available)
                values.append(trans.get("value", 0.0))
            
            if len(states) == 0:
                continue
            
            # Initialize networks if needed
            state_dim = len(states[0])
            if self.state_dim is None:
                self.state_dim = state_dim
            self._initialize_agent_networks(agent_name, state_dim)
            
            # Convert to tensors
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            rewards_array = np.array(rewards, dtype=np.float32)
            next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones_array = np.array(dones, dtype=bool)
            
            # Compute value estimates
            with torch.no_grad():
                current_values = self.values[agent_name](states_tensor).squeeze().cpu().numpy()
                next_values = self.values[agent_name](next_states_tensor).squeeze().cpu().numpy()
            
            # Compute GAE and returns
            advantages, returns = self._compute_gae(
                rewards_array.tolist(),
                current_values.tolist(),
                next_values.tolist(),
                dones_array.tolist()
            )
            
            returns_tensor = torch.FloatTensor(returns).to(self.device)
            advantages_tensor = torch.FloatTensor(advantages).to(self.device)
            
            # Update old policy
            self.old_policies[agent_name].load_state_dict(
                self.policies[agent_name].state_dict()
            )
            self.old_policies[agent_name].eval()
            
            # Training epochs
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            
            # Initialize accumulators for additional metrics
            self._value_estimates = []
            self._advantage_estimates = []
            self._policy_ratios = []
            self._kl_divergences = []
            self._policy_grad_norms = []
            self._value_grad_norms = []
            
            indices = np.arange(len(states))
            
            for epoch in range(self.num_epochs):
                np.random.shuffle(indices)
                
                # Mini-batch training
                for start in range(0, len(indices), self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    
                    batch_states = states_tensor[batch_indices]
                    batch_actions = actions_tensor[batch_indices]
                    batch_advantages = advantages_tensor[batch_indices]
                    batch_returns = returns_tensor[batch_indices]
                    
                    # Policy loss (PPO clipped surrogate)
                    # Note: PPO uses CLIP mechanism, not KL penalty
                    policy_dist = self.policies[agent_name](batch_states)
                    new_log_probs = policy_dist.log_prob(batch_actions)
                    
                    with torch.no_grad():
                        old_policy_dist = self.old_policies[agent_name](batch_states)
                        old_log_probs = old_policy_dist.log_prob(batch_actions)
                    
                    # Compute importance sampling ratio
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    
                    # PPO clipped surrogate objective
                    # L^CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Entropy bonus
                    entropy = policy_dist.entropy().mean()
                    entropy_bonus = -self.entropy_coef * entropy
                    
                    # Total policy loss
                    total_policy_loss_batch = policy_loss + entropy_bonus
                    
                    # Value loss
                    value_pred = self.values[agent_name](batch_states).squeeze()
                    value_loss = nn.functional.mse_loss(value_pred, batch_returns)
                    total_value_loss_batch = self.value_loss_coef * value_loss
                    
                    # Update policy
                    self.policy_optimizers[agent_name].zero_grad()
                    total_policy_loss_batch.backward()
                    # Compute gradient norm before clipping
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policies[agent_name].parameters(),
                        self.max_grad_norm
                    )
                    if not hasattr(self, '_policy_grad_norms'):
                        self._policy_grad_norms = []
                    self._policy_grad_norms.append(policy_grad_norm.item())
                    self.policy_optimizers[agent_name].step()
                    
                    # Update value
                    self.value_optimizers[agent_name].zero_grad()
                    total_value_loss_batch.backward()
                    # Compute gradient norm before clipping
                    value_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.values[agent_name].parameters(),
                        self.max_grad_norm
                    )
                    if not hasattr(self, '_value_grad_norms'):
                        self._value_grad_norms = []
                    self._value_grad_norms.append(value_grad_norm.item())
                    self.value_optimizers[agent_name].step()
                    
                    # Accumulate losses
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                    
                    # Accumulate additional metrics for visualization
                    # Value estimates
                    batch_values = value_pred.detach().cpu().numpy()
                    if not hasattr(self, '_value_estimates'):
                        self._value_estimates = []
                    self._value_estimates.extend(batch_values.tolist())
                    
                    # Advantage estimates
                    batch_advs = batch_advantages.detach().cpu().numpy()
                    if not hasattr(self, '_advantage_estimates'):
                        self._advantage_estimates = []
                    self._advantage_estimates.extend(batch_advs.tolist())
                    
                    # Policy ratios
                    batch_ratios = ratio.detach().cpu().numpy()
                    if not hasattr(self, '_policy_ratios'):
                        self._policy_ratios = []
                    self._policy_ratios.extend(batch_ratios.tolist())
                    
                    # KL divergence: KL(old || new) = E_old[log(old) - log(new)]
                    # Note: This is a MONITORING metric only. PPO uses CLIP mechanism, not KL penalty.
                    # We compute KL divergence to track how much the policy changes between updates.
                    with torch.no_grad():
                        # Compute KL divergence using torch.distributions
                        # KL(old || new) = E_old[log(old) - log(new)]
                        # For Categorical distributions:
                        try:
                            kl_div = torch.distributions.kl.kl_divergence(
                                old_policy_dist, policy_dist
                            ).mean().item()
                        except Exception:
                            # Fallback: approximate KL using log probs
                            # KL(old || new) ≈ E_old[log(old) - log(new)]
                            kl_div = (old_log_probs - new_log_probs).mean().item()
                    
                    if not hasattr(self, '_kl_divergences'):
                        self._kl_divergences = []
                    self._kl_divergences.append(kl_div)
                    
                    # Gradient norms (will be computed after backward)
                    if not hasattr(self, '_policy_grad_norms'):
                        self._policy_grad_norms = []
                        self._value_grad_norms = []
            
            # Average losses
            num_batches = (len(indices) + self.batch_size - 1) // self.batch_size
            avg_policy_loss = total_policy_loss / (self.num_epochs * num_batches)
            avg_value_loss = total_value_loss / (self.num_epochs * num_batches)
            avg_entropy = total_entropy / (self.num_epochs * num_batches)
            
            # Record training losses
            if self.metrics_recorder:
                self.metrics_recorder.record_training_loss(
                    agent_name=agent_name,
                    step=self.training_step,
                    policy_loss=avg_policy_loss,
                    value_loss=avg_value_loss,
                    entropy=avg_entropy
                )
                
                # Record additional metrics
                mean_value = np.mean(self._value_estimates) if hasattr(self, '_value_estimates') and self._value_estimates else None
                std_value = np.std(self._value_estimates) if hasattr(self, '_value_estimates') and len(self._value_estimates) > 1 else None
                
                mean_advantage = np.mean(self._advantage_estimates) if hasattr(self, '_advantage_estimates') and self._advantage_estimates else None
                std_advantage = np.std(self._advantage_estimates) if hasattr(self, '_advantage_estimates') and len(self._advantage_estimates) > 1 else None
                
                mean_kl = np.mean(self._kl_divergences) if hasattr(self, '_kl_divergences') and self._kl_divergences else None
                
                mean_policy_grad = np.mean(self._policy_grad_norms) if hasattr(self, '_policy_grad_norms') and self._policy_grad_norms else None
                mean_value_grad = np.mean(self._value_grad_norms) if hasattr(self, '_value_grad_norms') and self._value_grad_norms else None
                
                mean_ratio = np.mean(self._policy_ratios) if hasattr(self, '_policy_ratios') and self._policy_ratios else None
                std_ratio = np.std(self._policy_ratios) if hasattr(self, '_policy_ratios') and len(self._policy_ratios) > 1 else None
                # Calculate clip fraction (ratios outside [1-epsilon, 1+epsilon])
                clip_fraction = None
                if hasattr(self, '_policy_ratios') and self._policy_ratios:
                    clipped = sum(1 for r in self._policy_ratios if r < (1 - self.clip_epsilon) or r > (1 + self.clip_epsilon))
                    clip_fraction = clipped / len(self._policy_ratios)
                
                self.metrics_recorder.record_training_metrics(
                    agent_name=agent_name,
                    step=self.training_step,
                    mean_value=mean_value,
                    std_value=std_value,
                    mean_advantage=mean_advantage,
                    std_advantage=std_advantage,
                    kl_divergence=mean_kl,
                    policy_grad_norm=mean_policy_grad,
                    value_grad_norm=mean_value_grad,
                    mean_ratio=mean_ratio,
                    std_ratio=std_ratio,
                    clip_fraction=clip_fraction
                )
                
                # Reset accumulators for next training step
                if hasattr(self, '_value_estimates'):
                    del self._value_estimates
                if hasattr(self, '_advantage_estimates'):
                    del self._advantage_estimates
                if hasattr(self, '_policy_ratios'):
                    del self._policy_ratios
                if hasattr(self, '_kl_divergences'):
                    del self._kl_divergences
                if hasattr(self, '_policy_grad_norms'):
                    del self._policy_grad_norms
                if hasattr(self, '_value_grad_norms'):
                    del self._value_grad_norms
            
            # Diagnostic information
            max_entropy = np.log(self.action_dim)  # Theoretical max entropy for uniform distribution
            entropy_ratio = avg_entropy / max_entropy if max_entropy > 0 else 0.0
            
            print(f"Agent {agent_name} - Step {self.training_step}: "
                  f"Policy Loss: {avg_policy_loss:.4f}, "
                  f"Value Loss: {avg_value_loss:.4f}, "
                  f"Entropy: {avg_entropy:.4f} ({entropy_ratio*100:.1f}% of max={max_entropy:.2f}), "
                  f"Transitions: {len(transitions)}")
            
            # Warning if entropy is too low (less than 10% of max)
            if entropy_ratio < 0.1 and self.training_step > 5:
                print(f"  ⚠️  Warning: {agent_name}'s entropy is very low ({entropy_ratio*100:.1f}% of max). "
                      f"Policy may be too deterministic. Consider increasing entropy_coef.")
    
    def get_action(self, agent_name: str, state: Dict) -> Tuple[int, float, float]:
        """
        Get action from policy network
        
        Args:
            agent_name: Name of the agent
            state: State dictionary
            
        Returns:
            action_id: Action ID
            log_prob: Log probability of the action
            value: Value estimate
        """
        if agent_name not in self.policies:
            # Return default action if policy not initialized
            return 0, 0.0, 0.0
        
        # Extract state vector
        state_vec = self._extract_state_vector(state)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        
        # Get action from policy
        with torch.no_grad():
            policy_dist = self.policies[agent_name](state_tensor)
            action = policy_dist.sample()
            log_prob = policy_dist.log_prob(action)
            
            # Get value estimate
            value = self.values[agent_name](state_tensor).squeeze()
        
        return action.item(), log_prob.item(), value.item()
    
    def save_models(self, save_dir: str):
        """
        Save all agent models
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for agent_name in self.policies.keys():
            policy_path = os.path.join(save_dir, f"{agent_name}_policy.pt")
            value_path = os.path.join(save_dir, f"{agent_name}_value.pt")
            
            torch.save(self.policies[agent_name].state_dict(), policy_path)
            torch.save(self.values[agent_name].state_dict(), value_path)
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, load_dir: str):
        """
        Load all agent models
        
        Args:
            load_dir: Directory to load models from
        """
        for agent_name in self.policies.keys():
            policy_path = os.path.join(load_dir, f"{agent_name}_policy.pt")
            value_path = os.path.join(load_dir, f"{agent_name}_value.pt")
            
            if os.path.exists(policy_path):
                self.policies[agent_name].load_state_dict(torch.load(policy_path))
            if os.path.exists(value_path):
                self.values[agent_name].load_state_dict(torch.load(value_path))
        
        print(f"Models loaded from {load_dir}")

