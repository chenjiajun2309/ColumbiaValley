"""State extraction for RL training"""

import numpy as np
from typing import Dict, List, Any
from modules.agent import Agent
from modules.game import Game


class StateExtractor:
    """Extract state representation from Agent and Game for RL"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.max_memory_items = self.config.get("max_memory_items", 5)
        self.max_nearby_agents = self.config.get("max_nearby_agents", 3)
    
    def extract(self, agent: Agent, game: Game) -> Dict[str, Any]:
        """
        Extract state features from agent and game environment
        
        Args:
            agent: The agent to extract state for
            game: The game environment
            
        Returns:
            Dictionary containing state features
        """
        state = {
            "agent_id": agent.name,
            "persona_features": self._extract_persona_features(agent),
            "spatial_features": self._extract_spatial_features(agent),
            "action_features": self._extract_action_features(agent),
            "memory_features": self._extract_memory_features(agent),
            "social_features": self._extract_social_features(agent, game),
            "schedule_features": self._extract_schedule_features(agent),
        }
        return state
    
    def _extract_persona_features(self, agent: Agent) -> Dict:
        """Extract persona-related features"""
        persona = agent.scratch.scratch if hasattr(agent.scratch, 'scratch') else {}
        return {
            "currently": agent.scratch.currently,
            "persona_summary": str(persona)[:200] if persona else "",  
        }
    
    def _extract_spatial_features(self, agent: Agent) -> Dict:
        """Extract spatial/location features"""
        return {
            "coord": list(agent.coord) if agent.coord else [0, 0],
            "current_address": ":".join(agent.get_tile().get_address()[:3]) if agent.coord else "",
            "has_path": len(agent.path) > 0 if agent.path else False,
        }
    
    def _extract_action_features(self, agent: Agent) -> Dict:
        """Extract current action features"""
        if not agent.action:
            return {
                "action_type": "idle",
                "action_describe": "",
                "action_duration": 0,
                "action_progress": 0.0,
            }
        
        action_abstract = agent.action.abstract()
        event = agent.get_event()
        return {
            "action_type": event.predicate if event else "idle",
            "action_describe": event.get_describe()[:100] if event else "",
            "action_duration": agent.action.duration,
            "action_progress": self._compute_action_progress(agent),
        }
    
    def _extract_memory_features(self, agent: Agent) -> Dict:
        """Extract memory-related features"""
        concepts = agent.concepts[:self.max_memory_items]
        return {
            "num_concepts": len(agent.concepts),
            "recent_concepts": [c.describe[:50] for c in concepts],
            "poignancy": agent.status.get("poignancy", 0),
            "memory_size": agent.associate.index.nodes_num,
        }
    
    def _extract_social_features(self, agent: Agent, game: Game) -> Dict:
        """Extract social interaction features"""
        nearby_agents = self._get_nearby_agents(agent, game)
        relationships = {}
        
        for other_name, other_agent in nearby_agents.items():

            relationship_score = self._compute_relationship_score(agent, other_agent)
            relationships[other_name] = {
                "distance": self._compute_distance(agent.coord, other_agent.coord),
                "relationship_score": relationship_score,
                "recent_chats": len([c for c in agent.chats if c[0] == other_name]),
            }
        
        return {
            "nearby_agents": list(nearby_agents.keys())[:self.max_nearby_agents],
            "relationships": relationships,
            "num_nearby": len(nearby_agents),
        }
    
    def _extract_schedule_features(self, agent: Agent) -> Dict:
        """Extract schedule-related features"""
        if not agent.schedule.scheduled():
            return {
                "has_schedule": False,
                "current_plan": "",
                "next_plan": "",
            }
        
        plan, _ = agent.schedule.current_plan()
        return {
            "has_schedule": True,
            "current_plan": plan.get("describe", "")[:50],
            "schedule_progress": self._compute_schedule_progress(agent),
        }
    
    def _get_nearby_agents(self, agent: Agent, game: Game, max_distance: int = 5) -> Dict:
        """Get agents nearby the current agent"""
        nearby = {}
        if not agent.coord:
            return nearby
        
        for name, other in game.agents.items():
            if name == agent.name or not other.coord:
                continue
            distance = self._compute_distance(agent.coord, other.coord)
            if distance <= max_distance:
                nearby[name] = other
        
        return nearby
    
    def _compute_distance(self, coord1, coord2) -> float:
        """Compute Euclidean distance between two coordinates"""
        if not coord1 or not coord2:
            return float('inf')
        return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5
    
    def _compute_relationship_score(self, agent: Agent, other: Agent) -> float:
        """Compute relationship score between two agents"""

        chats = agent.associate.retrieve_chats(other.name)
        if not chats:
            return 0.0
        

        recent_chats = [c for c in chats if c.access]
        if not recent_chats:
            return 0.0
        

        return sum(c.poignancy for c in recent_chats[:5]) / 10.0
    
    def _compute_action_progress(self, agent: Agent) -> float:
        """Compute progress of current action (0.0 to 1.0)"""
        if not agent.action or not agent.action.duration:
            return 1.0
        
        from modules import utils
        elapsed = (utils.get_timer().get_date() - agent.action.start).total_seconds() / 60.0
        return min(elapsed / agent.action.duration, 1.0)
    
    def _compute_schedule_progress(self, agent: Agent) -> float:
        """Compute progress through daily schedule"""
        if not agent.schedule.scheduled():
            return 0.0
        
        from modules import utils
        current_time = utils.get_timer().daily_duration()

        return current_time / (24 * 60)
    
    def state_to_vector(self, state: Dict) -> np.ndarray:
        """
        Convert state dictionary to vector representation for neural network
        
        This is a simplified version. In practice, you might want to use
        more sophisticated encoding (e.g., embeddings for text features)
        """

        features = []
        

        features.append(len(state["persona_features"]["currently"]))
        
        # Spatial features
        features.extend(state["spatial_features"]["coord"])
        features.append(1.0 if state["spatial_features"]["has_path"] else 0.0)
        
        # Action features
        features.append(state["action_features"]["action_duration"])
        features.append(state["action_features"]["action_progress"])
        
        # Memory features
        features.append(state["memory_features"]["num_concepts"])
        features.append(state["memory_features"]["poignancy"])
        
        # Social features
        features.append(state["social_features"]["num_nearby"])
        
        # Schedule features
        features.append(1.0 if state["schedule_features"]["has_schedule"] else 0.0)
        
        return np.array(features, dtype=np.float32)

