"""Reward function for RL training"""

from typing import Dict, Any, List
import numpy as np
from modules.agent import Agent
from modules.game import Game
from modules.rl.action_space import ActionType


class RewardFunction:
    """Compute multi-objective rewards for RL training"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        # Reward weights（可以通过 config 覆盖）
        # 默认为：社交质量 + 关系提升 为主，日程与多样性为辅，
        # persona_alignment 用作「软约束」，提醒 RL 不要过度偏离角色性格。
        self.weights = {
            "persona_alignment": self.config.get("persona_alignment_weight", 0.15),
            "interaction_quality": self.config.get("interaction_quality_weight", 0.4),
            "relationship_growth": self.config.get("relationship_growth_weight", 0.3),
            "diversity": self.config.get("diversity_weight", 0.15),
            "schedule_completion": self.config.get("schedule_completion_weight", 0.15),
        }
    
    def compute_reward(
        self,
        agent: Agent,
        action_dict: Dict,
        next_state: Dict,
        game: Game,
        previous_state: Dict = None,
        return_components: bool = False
    ):
        """
        Compute reward for agent's action
        
        Args:
            agent: The agent that took the action
            action_dict: The action that was taken
            next_state: State after taking the action
            game: The game environment
            previous_state: State before taking the action (optional)
            return_components: If True, return (reward, components_dict)
            
        Returns:
            Reward value (float) or (reward, components_dict) if return_components=True
        """
        reward = 0.0
        components = {}
        
        # 1. Persona alignment reward
        persona_reward = self._compute_persona_alignment(agent, action_dict)
        persona_contribution = self.weights["persona_alignment"] * persona_reward
        reward += persona_contribution
        components["persona_alignment"] = persona_reward
        components["persona_alignment_weighted"] = persona_contribution
        
        # 2. Interaction quality reward
        interaction_reward = self._compute_interaction_quality(agent, action_dict, game)
        interaction_contribution = self.weights["interaction_quality"] * interaction_reward
        reward += interaction_contribution
        components["interaction_quality"] = interaction_reward
        components["interaction_quality_weighted"] = interaction_contribution
        
        # 3. Relationship growth reward
        relationship_reward = self._compute_relationship_growth(
            agent, action_dict, game, previous_state, next_state
        )
        relationship_contribution = self.weights["relationship_growth"] * relationship_reward
        reward += relationship_contribution
        components["relationship_growth"] = relationship_reward
        components["relationship_growth_weighted"] = relationship_contribution
        
        # 4. Diversity reward (avoid repetitive behavior)
        diversity_reward = self._compute_diversity_bonus(agent, action_dict)
        diversity_contribution = self.weights["diversity"] * diversity_reward
        reward += diversity_contribution
        components["diversity"] = diversity_reward
        components["diversity_weighted"] = diversity_contribution
        
        # 5. Schedule completion reward
        schedule_reward = self._compute_schedule_completion(agent)
        schedule_contribution = self.weights["schedule_completion"] * schedule_reward
        reward += schedule_contribution
        components["schedule_completion"] = schedule_reward
        components["schedule_completion_weighted"] = schedule_contribution
        
        components["total"] = reward
        
        if return_components:
            return reward, components
        return reward
    
    def _compute_persona_alignment(self, agent: Agent, action_dict: Dict) -> float:
        """
        Compute reward for persona alignment
        
        Returns value between -1 and 1
        """
        # 一个轻量级的、基于关键词的 persona 对齐奖励：
        # - 不额外调用 LLM，避免在每一步 RL 里增加开销；
        # - 只作为「软偏好」，幅度在 [-1, 1]，由外层权重缩放。
        action_type = action_dict["type"]

        # 从 Scratch 中拿到 persona 文本
        scratch_cfg = getattr(agent.scratch, "config", {}) or {}
        innate = str(scratch_cfg.get("innate", "")).lower()
        learned = str(scratch_cfg.get("learned", "")).lower()
        lifestyle = str(scratch_cfg.get("lifestyle", "")).lower()
        persona_text = " ".join([innate, learned, lifestyle])

        score = 0.0

        # 社交型：喜欢与人互动
        social_keywords = ["outgoing", "social", "collaborative", "friendly", "helpful", "talkative"]
        is_social = any(k in persona_text for k in social_keywords)
        if is_social:
            if action_type == ActionType.INITIATE_CHAT:
                score += 0.8
            elif action_type == ActionType.WAIT:
                score += 0.2
            elif action_type == ActionType.SKIP_REACTION:
                score -= 0.6

        # 探索 / 好奇：更愿意移动、更改计划
        curious_keywords = ["curious", "explor", "adventurous", "open-minded"]
        is_curious = any(k in persona_text for k in curious_keywords)
        if is_curious:
            if action_type in (ActionType.CHANGE_LOCATION, ActionType.REVISE_SCHEDULE):
                score += 0.6
            elif action_type == ActionType.WAIT:
                score -= 0.2

        # 沉稳 / 有条理：更偏向按计划执行、少改动
        disciplined_keywords = ["methodical", "organized", "disciplined", "calm", "focused"]
        is_disciplined = any(k in persona_text for k in disciplined_keywords)
        if is_disciplined:
            if action_type == ActionType.CONTINUE:
                score += 0.6
            elif action_type in (ActionType.CHANGE_LOCATION, ActionType.REVISE_SCHEDULE):
                score -= 0.3

        # 将得分裁剪到 [-1, 1]
        return float(np.clip(score, -1.0, 1.0))
    
    def _compute_interaction_quality(self, agent: Agent, action_dict: Dict, game: Game) -> float:
        """
        Compute reward for interaction quality
        
        Returns value between -1 and 1
        """
        action_type = action_dict["type"]
        
        if action_type == ActionType.INITIATE_CHAT:
            target_name = action_dict.get("target_agent")
            if not target_name or target_name not in game.agents:
                return -0.5  # Penalty for invalid target
            
            target = game.agents[target_name]
            
            # Check if chat was successful (simplified)
            # In practice, you'd check conversation logs
            recent_chats = [c for c in agent.chats if c[0] == target_name]
            if recent_chats:
                # 根据最近一次对话长度给奖励，限制在 [0, 0.8] 左右，
                # 留出一部分空间给 relationship_growth。
                return min(len(recent_chats[-1][1]) / 120.0, 0.8)
            else:
                # 发起了聊天但没有记录（极大可能失败）给一个中等偏小的惩罚，
                # 防止 agent 疯狂乱聊，但又不至于完全不探索聊天。
                return -0.1
        
        elif action_type == ActionType.WAIT:
            # WAIT 本身不给强正奖励，避免 agent 通过高频等待「刷分」。
            # 只有当后续聊天/关系提升时，才通过其他 reward 体现收益。
            return 0.02  # 轻微正奖励，鼓励合理耐心，而不是主要得分手段
        
        return 0.0
    
    def _compute_relationship_growth(
        self,
        agent: Agent,
        action_dict: Dict,
        game: Game,
        previous_state: Dict,
        next_state: Dict
    ) -> float:
        """
        Compute reward for relationship growth
        
        Returns value between -1 and 1
        """
        if not previous_state or not next_state:
            return 0.0
        
        action_type = action_dict["type"]
        
        if action_type == ActionType.INITIATE_CHAT:
            target_name = action_dict.get("target_agent")
            if not target_name:
                return 0.0
            
            # Check relationship score change
            prev_rel = previous_state.get("social_features", {}).get("relationships", {}).get(
                target_name, {}
            ).get("relationship_score", 0.0)
            
            next_rel = next_state.get("social_features", {}).get("relationships", {}).get(
                target_name, {}
            ).get("relationship_score", 0.0)
            
            # Reward relationship improvement.
            # 关系分数通常变化很小，这里适当放大系数，确保有可见信号。
            growth = next_rel - prev_rel
            # 正向幅度稍大，负向削弱，鼓励尝试社交即便偶尔失败。
            if growth >= 0:
                return float(np.clip(growth * 4.0, 0.0, 1.0))
            else:
                return float(np.clip(growth * 2.0, -1.0, 0.0))
        
        return 0.0
    
    def _compute_diversity_bonus(self, agent: Agent, action_dict: Dict) -> float:
        """
        Compute reward for behavioral diversity
        
        Returns value between -1 and 1
        """
        # Check action history to encourage diversity
        if not hasattr(agent, 'rl_action_history'):
            agent.rl_action_history = []
        
        action_type = action_dict["type"]
        
        # 我们只对「低价值行为」的过度重复进行惩罚，避免和按日程持续行动的目标冲突。
        # 目前认为 CONTINUE / WAIT 属于此类，如果高价值动作重复，则不额外处罚。
        low_value_actions = {ActionType.CONTINUE, ActionType.WAIT}

        if len(agent.rl_action_history) > 0 and action_type in low_value_actions:
            recent_actions = agent.rl_action_history[-5:]  # Last 5 actions
            same_action_count = sum(1 for a in recent_actions if a == action_type)

            # 最近 5 步里几乎全是同一低价值动作 → 轻微惩罚
            if same_action_count >= 4:
                return -0.15
            # 完全没出现过该动作 → 鼓励探索新类型
            elif same_action_count == 0:
                return 0.1
        
        # Update history
        agent.rl_action_history.append(action_type)
        if len(agent.rl_action_history) > 20:
            agent.rl_action_history = agent.rl_action_history[-20:]
        
        return 0.0
    
    def _compute_schedule_completion(self, agent: Agent) -> float:
        """
        Compute reward for schedule completion
        
        Returns value between -1 and 1
        """
        if not agent.schedule.scheduled():
            return 0.0
        
        plan, _ = agent.schedule.current_plan()
        current_action = agent.get_event().get_describe()
        planned_action = plan.get("describe", "")
        
        # Simple check: if current action matches planned action
        if planned_action.lower() in current_action.lower() or current_action.lower() in planned_action.lower():
            return 0.2  # Reward for following schedule
        else:
            return -0.1  # Small penalty for deviating
        
        return 0.0

