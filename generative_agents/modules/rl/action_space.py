"""Action space definition for RL agents"""

from enum import IntEnum
from typing import Dict, List, Any, Optional
from modules.agent import Agent
from modules.game import Game


class ActionType(IntEnum):
    """Action types for RL agent"""
    CONTINUE = 0           # Continue current action
    INITIATE_CHAT = 1      # Initiate conversation with nearby agent
    WAIT = 2              # Wait for other agent
    CHANGE_LOCATION = 3    # Change location (follow path)
    REVISE_SCHEDULE = 4    # Revise current schedule
    SKIP_REACTION = 5      # Skip reacting to nearby events


class ActionSpace:
    """Define and manage action space for RL agents"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.num_action_types = len(ActionType)
    
    def get_action_space_size(self) -> int:
        """Get the size of action space"""
        return self.num_action_types
    
    def decode_action(self, action_id: int, agent: Agent, game: Game) -> Dict[str, Any]:
        """
        Decode action ID to actionable parameters
        
        Args:
            action_id: The action ID from policy network
            agent: The agent taking the action
            game: The game environment
            
        Returns:
            Dictionary with action details
        """
        action_type = ActionType(action_id)
        
        action_dict = {
            "type": action_type,
            "action_id": action_id,
        }
        
        # Add action-specific parameters
        if action_type == ActionType.INITIATE_CHAT:
            # Select target agent
            target = self._select_chat_target(agent, game)
            action_dict["target_agent"] = target
        
        elif action_type == ActionType.CHANGE_LOCATION:
            # Use existing path or determine new location
            action_dict["use_existing_path"] = len(agent.path) > 0
        
        elif action_type == ActionType.REVISE_SCHEDULE:
            # Determine schedule revision
            action_dict["revision_type"] = "minor"  # Could be learned
        
        return action_dict
    
    def _select_chat_target(self, agent: Agent, game: Game) -> Optional[str]:
        """Select target agent for chat action"""
        # Find nearby agents
        nearby = []
        for name, other in game.agents.items():
            if name == agent.name or not other.coord:
                continue
            # Simple distance check
            if agent.coord and other.coord:
                dist = ((agent.coord[0] - other.coord[0])**2 + 
                       (agent.coord[1] - other.coord[1])**2)**0.5
                if dist <= 3:  # Within chat range
                    nearby.append((name, other, dist))
        
        if not nearby:
            return None
        
        # Select closest agent
        nearby.sort(key=lambda x: x[2])
        return nearby[0][0]
    
    def is_action_valid(self, action_dict: Dict, agent: Agent, game: Game) -> bool:
        """
        Check if an action is valid in current state
        
        Args:
            action_dict: The decoded action dictionary
            agent: The agent taking the action
            game: The game environment
            
        Returns:
            True if action is valid, False otherwise
        """
        action_type = action_dict["type"]
        
        if action_type == ActionType.INITIATE_CHAT:
            target = action_dict.get("target_agent")
            if not target or target not in game.agents:
                return False
            # Check if target is nearby and available
            target_agent = game.agents[target]
            if not target_agent.coord or not agent.coord:
                return False
            # Additional checks (e.g., not already chatting)
            if (agent.get_event().fit(predicate="conversation") or 
                target_agent.get_event().fit(predicate="conversation")):
                return False
        
        elif action_type == ActionType.CHANGE_LOCATION:
            # Check if there's a valid path or location to move to
            if not action_dict.get("use_existing_path") and not agent.path:
                # Would need to determine new location
                pass
        
        elif action_type == ActionType.WAIT:
            # Check if there's someone to wait for
            if not self._has_wait_target(agent, game):
                return False
        
        return True
    
    def _has_wait_target(self, agent: Agent, game: Game) -> bool:
        """Check if there's a valid target to wait for"""
        if not agent.path:
            return False
        
        # Check if path leads to another agent
        target_address = agent.get_event().address
        for name, other in game.agents.items():
            if name == agent.name:
                continue
            if other.get_tile().get_address() == target_address:
                return True
        return False
    
    def apply_action(self, action_dict: Dict, agent: Agent, game: Game) -> bool:
        """
        Apply action to agent (modify agent state)
        
        Note: This should be called carefully as it modifies agent state.
        In training, you might want to work with copies.
        
        Args:
            action_dict: The decoded action dictionary
            agent: The agent taking the action
            game: The game environment
            
        Returns:
            True if action was successfully applied
        """
        action_type = action_dict["type"]
        
        if not self.is_action_valid(action_dict, agent, game):
            return False
        
        # Store action for later execution
        # In practice, you might want to queue actions or apply them immediately
        agent.rl_action = action_dict
        
        return True

