"""
Online Data Collector for RL Training

This module collects state-action-reward tuples in real-time
as agents make decisions and interact with each other.
"""

from typing import Dict, List, Optional, Any
from collections import defaultdict
from modules.agent import Agent
from modules.game import Game
from modules.rl.state_extractor import StateExtractor
from modules.rl.action_space import ActionSpace, ActionType
from modules.rl.reward_function import RewardFunction
from modules.rl.metrics_recorder import RLMetricsRecorder


class OnlineDataCollector:
    """
    Collect RL training data in real-time during agent execution
    
    This collector hooks into Agent's decision-making process to collect:
    - States: Before actions are taken
    - Actions: Decisions made by agents (both LLM and RL)
    - Rewards: Computed after actions are executed
    - Next states: After actions are executed
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.state_extractor = StateExtractor(config.get("state_extractor", {}))
        self.action_space = ActionSpace(config.get("action_space", {}))
        self.reward_function = RewardFunction(config.get("reward_function", {}))
        
        # Data buffers for each agent
        self.rollout_buffers = defaultdict(list)  # {agent_name: [transitions]}
        
        # Track previous states for reward computation
        self.previous_states = {}  # {agent_name: state_dict}
        
        # Track actions taken
        self.current_actions = {}  # {agent_name: action_dict}
        
        # Enable/disable collection
        self.enabled = True
        
        # Metrics-only mode: if True, only record metrics, don't collect rollouts for training
        self.metrics_only = self.config.get("metrics_only", False)
        
        # Episode interval for baseline (metrics-only mode): end episode every N steps
        # This allows baseline experiments to have episode returns for comparison
        # Default: every 10 steps (similar to typical rl_train_interval)
        self.episode_interval = self.config.get("episode_interval", 10)
        
        # Maximum buffer size before flushing
        self.max_buffer_size = self.config.get("max_buffer_size", 1000)
        
        # Metrics recorder
        metrics_config = config.get("metrics", {})
        metrics_config["checkpoints_folder"] = config.get("checkpoints_folder", "")
        self.metrics_recorder = RLMetricsRecorder(metrics_config)
    
    def start_collection(self, agent_name: str, agent: Agent, game: Game):
        """
        Start collecting data for an agent (called before decision-making)
        
        This should be called at the beginning of Agent.think() or similar methods
        
        Args:
            agent_name: Name of the agent
            agent: Agent instance
            game: Game environment
        """
        if not self.enabled:
            return
        
        # Extract current state
        state = self.state_extractor.extract(agent, game)
        self.previous_states[agent_name] = state
    
    def record_llm_action(
        self,
        agent_name: str,
        agent: Agent,
        game: Game,
        action_type: str,
        action_details: Dict = None
    ):
        """
        Record an action taken by LLM (original decision-making)
        
        This should be called when LLM makes a decision, e.g.:
        - When deciding to chat (_chat_with)
        - When determining action (_determine_action)
        - When reacting (_reaction)
        
        Args:
            agent_name: Name of the agent
            agent: Agent instance
            game: Game environment
            action_type: Type of action (e.g., "chat", "move", "wait")
            action_details: Additional details about the action
        """
        if not self.enabled or agent_name not in self.previous_states:
            return
        
        # Map LLM action to RL action space
        rl_action_type = self._map_llm_action_to_rl(action_type, action_details)
        
        # Create action dict
        action_dict = {
            "type": rl_action_type,
            "action_id": int(rl_action_type),
            "source": "llm",  # Mark as LLM decision
            "llm_action_type": action_type,
            "details": action_details or {},
        }
        
        self.current_actions[agent_name] = action_dict
        
        # Record action in metrics
        self.metrics_recorder.record_action(agent_name, str(rl_action_type))
    
    def record_rl_action(
        self,
        agent_name: str,
        agent: Agent,
        game: Game,
        action_id: int,
        action_log_prob: float = None,
        value: float = None
    ):
        """
        Record an action taken by RL policy
        
        Args:
            agent_name: Name of the agent
            agent: Agent instance
            game: Game environment
            action_id: Action ID from RL policy
            action_log_prob: Log probability of the action
            value: Value estimate
        """
        if not self.enabled or agent_name not in self.previous_states:
            return
        
        # Decode action
        action_dict = self.action_space.decode_action(action_id, agent, game)
        action_dict["source"] = "rl"
        action_dict["action_log_prob"] = action_log_prob
        action_dict["value"] = value
        
        self.current_actions[agent_name] = action_dict
    
    def end_collection(
        self,
        agent_name: str,
        agent: Agent,
        game: Game,
        done: bool = False
    ):
        """
        End collection for an agent (called after action execution)
        
        This should be called at the end of Agent.think() or after actions are executed
        
        Args:
            agent_name: Name of the agent
            agent: Agent instance
            game: Game environment
            done: Whether the episode is done
        """
        if not self.enabled:
            return
        
        if agent_name not in self.previous_states:
            return
        
        # Extract next state
        next_state = self.state_extractor.extract(agent, game)
        
        # Get action that was taken
        action_dict = self.current_actions.get(agent_name)
        if not action_dict:
            # No explicit action recorded, create a default action based on agent's current state
            # This ensures we still record rewards even if no explicit action was recorded
            from modules.rl.action_space import ActionType
            default_action_type = ActionType.CONTINUE  # Default to continue current activity
            
            # Try to infer action from agent's current action
            if hasattr(agent, 'action') and agent.action:
                # Use str() to get string representation, as abstract() returns a dict
                action_desc = str(agent.action)
                # Check if it's a chat action
                if hasattr(agent, 'chats') and agent.chats:
                    default_action_type = ActionType.INITIATE_CHAT
                # Check if it's a move action
                elif "move" in action_desc.lower() or "go" in action_desc.lower():
                    default_action_type = ActionType.CHANGE_LOCATION
                # Check if it's a wait/sleep action
                elif "sleep" in action_desc.lower() or "wait" in action_desc.lower():
                    default_action_type = ActionType.WAIT
            
            action_dict = {
                "type": default_action_type,
                "action_id": int(default_action_type),
                "source": "llm",
                "llm_action_type": "default",
                "details": {"inferred": True, "action_desc": str(agent.action) if hasattr(agent, 'action') else "unknown"}
            }
            self.current_actions[agent_name] = action_dict
            # Record this default action in metrics
            self.metrics_recorder.record_action(agent_name, str(default_action_type))
        
        # Compute reward with components
        reward, components = self.reward_function.compute_reward(
            agent,
            action_dict,
            next_state,
            game,
            self.previous_states[agent_name],
            return_components=True
        )
        
        # Record reward in metrics
        # Use current training step if available, otherwise use 0
        current_step = self.metrics_recorder.training_step if hasattr(self.metrics_recorder, 'training_step') else 0
        self.metrics_recorder.record_reward(
            agent_name, reward, components, step=current_step
        )
        
        # Store transition only if not in metrics-only mode (baseline experiments don't need rollouts)
        if not self.metrics_only:
            transition = {
                "state": self.previous_states[agent_name],
                "action": action_dict["action_id"],
                "action_dict": action_dict,
                "action_log_prob": action_dict.get("action_log_prob"),
                "value": action_dict.get("value"),
                "reward": reward,
                "next_state": next_state,
                "done": done,
            }
            self.rollout_buffers[agent_name].append(transition)
        
        # Update previous state
        self.previous_states[agent_name] = next_state
        self.current_actions.pop(agent_name, None)
        
        # In metrics-only mode (baseline), end episode at fixed intervals
        # This allows baseline experiments to have episode returns for comparison
        if self.metrics_only:
            current_step = self.metrics_recorder.training_step if hasattr(self.metrics_recorder, 'training_step') else 0
            # End episode every episode_interval steps (similar to training intervals in RL mode)
            if current_step > 0 and current_step % self.episode_interval == 0:
                self.metrics_recorder.end_episode(current_step)
        
        # Flush buffer if it's too large
        if len(self.rollout_buffers[agent_name]) >= self.max_buffer_size:
            self.flush_buffer(agent_name)
    
    def record_chat_interaction(
        self,
        agent1_name: str,
        agent2_name: str,
        agent1: Agent,
        agent2: Agent,
        game: Game,
        chats: List,
        chat_summary: str
    ):
        """
        Record a chat interaction between two agents
        
        This should be called after a chat is completed in _chat_with()
        
        Args:
            agent1_name: Name of initiating agent
            agent2_name: Name of responding agent
            agent1: Initiating agent instance
            agent2: Responding agent instance
            game: Game environment
            chats: List of chat messages
            chat_summary: Summary of the chat
        """
        if not self.enabled:
            return
        
        # Record interaction reward for both agents
        # This is a special case where both agents participate
        
        # For agent1 (initiator)
        if agent1_name in self.previous_states:
            action_dict = {
                "type": ActionType.INITIATE_CHAT,
                "action_id": int(ActionType.INITIATE_CHAT),
                "source": "llm",
                "target_agent": agent2_name,
                "chats": chats,
                "chat_summary": chat_summary,
            }
            
            next_state = self.state_extractor.extract(agent1, game)
            reward, components = self.reward_function.compute_reward(
                agent1, action_dict, next_state, game, self.previous_states[agent1_name],
                return_components=True
            )
            
            # Record reward in metrics
            current_step = self.metrics_recorder.training_step if hasattr(self.metrics_recorder, 'training_step') else 0
            self.metrics_recorder.record_reward(agent1_name, reward, components, step=current_step)
            
            # Store transition only if not in metrics-only mode
            if not self.metrics_only:
                transition = {
                    "state": self.previous_states[agent1_name],
                    "action": int(ActionType.INITIATE_CHAT),
                    "action_dict": action_dict,
                    "reward": reward,
                    "next_state": next_state,
                    "done": False,
                }
                self.rollout_buffers[agent1_name].append(transition)
            self.previous_states[agent1_name] = next_state
        
        # For agent2 (responder) - similar but with different action type
        if agent2_name in self.previous_states:
            action_dict = {
                "type": ActionType.CONTINUE,  # Responding is continuing the interaction
                "action_id": int(ActionType.CONTINUE),
                "source": "llm",
                "interaction_type": "chat_response",
                "chats": chats,
                "chat_summary": chat_summary,
            }
            
            next_state = self.state_extractor.extract(agent2, game)
            reward, components = self.reward_function.compute_reward(
                agent2, action_dict, next_state, game, self.previous_states[agent2_name],
                return_components=True
            )
            
            # Record reward in metrics
            current_step = self.metrics_recorder.training_step if hasattr(self.metrics_recorder, 'training_step') else 0
            self.metrics_recorder.record_reward(agent2_name, reward, components, step=current_step)
            
            # Store transition only if not in metrics-only mode
            if not self.metrics_only:
                transition = {
                    "state": self.previous_states[agent2_name],
                    "action": int(ActionType.CONTINUE),
                    "action_dict": action_dict,
                    "reward": reward,
                    "next_state": next_state,
                    "done": False,
                }
                self.rollout_buffers[agent2_name].append(transition)
            self.previous_states[agent2_name] = next_state
    
    def _map_llm_action_to_rl(self, action_type: str, details: Dict = None) -> ActionType:
        """
        Map LLM action type to RL action space
        
        Args:
            action_type: LLM action type (e.g., "chat", "move")
            details: Additional details
            
        Returns:
            Corresponding RL ActionType
        """
        mapping = {
            "chat": ActionType.INITIATE_CHAT,
            "conversation": ActionType.INITIATE_CHAT,
            "wait": ActionType.WAIT,
            "move": ActionType.CHANGE_LOCATION,
            "determine_action": ActionType.CONTINUE,
            "think": ActionType.CONTINUE,  # Thinking/planning is continuing current activity
            "sleep": ActionType.WAIT,  # Sleeping is a form of waiting
            "revise_schedule": ActionType.REVISE_SCHEDULE,
            "skip": ActionType.SKIP_REACTION,
        }
        
        return mapping.get(action_type.lower(), ActionType.CONTINUE)
    
    def get_rollouts(self, agent_names: List[str] = None) -> Dict[str, List]:
        """
        Get collected rollouts for specified agents (or all agents)
        
        Args:
            agent_names: List of agent names, or None for all
            
        Returns:
            Dictionary of rollouts per agent
        """
        if agent_names is None:
            return dict(self.rollout_buffers)
        return {name: self.rollout_buffers.get(name, []) for name in agent_names}
    
    def flush_buffer(self, agent_name: str) -> List:
        """
        Flush and return buffer for an agent
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of transitions
        """
        if agent_name not in self.rollout_buffers:
            return []
        
        transitions = self.rollout_buffers[agent_name]
        self.rollout_buffers[agent_name] = []
        return transitions
    
    def flush_all_buffers(self) -> Dict[str, List]:
        """
        Flush all buffers and return all collected data
        
        This is typically called at training intervals to:
        1. Get all collected rollouts for training
        2. End current episode and record episode returns
        
        Returns:
            Dictionary of all rollouts
        """
        # End current episode before flushing (episode = sequence between training intervals)
        # This is only for RL mode; baseline mode uses update_step() to end episodes
        if not self.metrics_only:
            current_step = self.metrics_recorder.training_step if hasattr(self.metrics_recorder, 'training_step') else 0
            self.metrics_recorder.end_episode(current_step)
        
        all_rollouts = dict(self.rollout_buffers)
        self.rollout_buffers.clear()
        self.previous_states.clear()
        self.current_actions.clear()
        return all_rollouts
    
    def clear(self):
        """Clear all collected data"""
        self.rollout_buffers.clear()
        self.previous_states.clear()
        self.current_actions.clear()
    
    def enable(self):
        """Enable data collection"""
        self.enabled = True
    
    def disable(self):
        """Disable data collection"""
        self.enabled = False

