"""
Integration Example: How to integrate OnlineDataCollector into Agent and Game

This file shows how to modify Agent.think() and related methods to collect
RL training data in real-time.
"""

# ============================================================================
# Example 1: Modify Agent.think() to collect data
# ============================================================================

def agent_think_with_collection(self, status, agents):
    """
    Modified Agent.think() with data collection hooks
    
    Add this to Agent class or modify existing think() method
    """
    from modules.rl.data_collector import OnlineDataCollector
    from modules.game import get_game
    
    game = get_game()
    collector = getattr(game, 'rl_collector', None)
    
    # 1. Start collection (before decision-making)
    if collector:
        collector.start_collection(self.name, self, game)
    
    # Original think logic
    events = self.move(status["coord"], status.get("path"))
    plan, _ = self.make_schedule()
    
    # ... existing code ...
    
    if self.is_awake():
        self.percept()
        self.make_plan(agents)
        self.reflect()
    else:
        if self.action.finished():
            self.action = self._determine_action()
            # 2. Record action decision
            if collector:
                collector.record_llm_action(
                    self.name, self, game, "determine_action",
                    {"action": self.action.abstract()}
                )
    
    # ... existing code ...
    
    # 3. End collection (after action execution)
    if collector:
        collector.end_collection(self.name, self, game, done=False)
    
    return self.plan


# ============================================================================
# Example 2: Modify Agent._chat_with() to collect chat interactions
# ============================================================================

def agent_chat_with_collection(self, other, focus):
    """
    Modified Agent._chat_with() with data collection
    
    Add this to Agent class or modify existing _chat_with() method
    """
    from modules.rl.data_collector import OnlineDataCollector
    from modules.game import get_game
    
    game = get_game()
    collector = getattr(game, 'rl_collector', None)
    
    # Record that we're initiating a chat
    if collector:
        collector.record_llm_action(
            self.name, self, game, "chat",
            {"target": other.name, "focus": focus.describe if focus else None}
        )
    
    # Original chat logic
    if not self.completion("decide_chat", self, other, focus, chats):
        return False
    
    # ... existing chat generation code ...
    
    # After chat is completed
    chat_summary = self.completion("summarize_chats", chats)
    duration = int(sum([len(c[1]) for c in chats]) / 240)
    
    # Record the chat interaction
    if collector:
        collector.record_chat_interaction(
            self.name, other.name, self, other, game,
            chats, chat_summary
        )
    
    self.schedule_chat(chats, chat_summary, start, duration, other)
    other.schedule_chat(chats, chat_summary, start, duration, self)
    
    return True


# ============================================================================
# Example 3: Modify Game.agent_think() to integrate collector
# ============================================================================

def game_agent_think_with_collection(self, name, status):
    """
    Modified Game.agent_think() with data collection
    
    Add this to Game class or modify existing agent_think() method
    """
    agent = self.get_agent(name)
    
    # Agent.think() will handle collection internally
    plan = agent.think(status, self.agents)
    
    # ... existing code ...
    
    return {"plan": plan, "info": info}


# ============================================================================
# Example 4: Initialize collector in Game.__init__()
# ============================================================================

def game_init_with_collector(self, name, static_root, config, conversation, logger=None):
    """
    Modified Game.__init__() to initialize data collector
    
    Add this to Game class or modify existing __init__() method
    """
    # ... existing initialization code ...
    
    # Initialize RL data collector if RL is enabled
    if config.get("use_rl", False):
        from modules.rl.data_collector import OnlineDataCollector
        rl_config = config.get("rl", {})
        self.rl_collector = OnlineDataCollector(rl_config.get("collector", {}))
    else:
        self.rl_collector = None
    
    # ... rest of initialization ...


# ============================================================================
# Example 5: Training loop with online collection
# ============================================================================

def train_with_online_collection(game, trainer, num_episodes=100, steps_per_episode=100):
    """
    Training loop that uses online data collection
    
    Args:
        game: Game instance with rl_collector
        trainer: MAPPOTrainer instance
        num_episodes: Number of training episodes
        steps_per_episode: Steps per episode
    """
    from modules import utils
    
    for episode in range(num_episodes):
        # Clear collector at start of episode
        if game.rl_collector:
            game.rl_collector.clear()
        
        # Run simulation (this will collect data automatically)
        for step in range(steps_per_episode):
            # Simulate one step for all agents
            for name, status in game.agent_status.items():
                game.agent_think(name, status)
            
            # Advance time
            # utils.get_timer().forward(stride)
        
        # Collect rollouts from collector
        if game.rl_collector:
            rollouts = game.rl_collector.flush_all_buffers()
            
            # Train on collected data
            if rollouts and any(len(r) > 0 for r in rollouts.values()):
                trainer.train_step(rollouts)
        
        # Save checkpoint periodically
        if episode % 10 == 0:
            trainer.save_models(f"checkpoints/episode_{episode}")
            print(f"Episode {episode} completed, models saved")


# ============================================================================
# Example 6: Alternative - Collect from checkpoints (OFFLINE)
# ============================================================================

def collect_from_checkpoints(checkpoint_path: str, agent_names: List[str]):
    """
    Alternative approach: Collect data from saved checkpoints
    
    This is for OFFLINE learning, not online RL.
    Use this if you want to train on historical data.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        agent_names: List of agent names
    """
    import json
    import os
    from modules.rl.state_extractor import StateExtractor
    from modules.rl.action_space import ActionSpace
    
    state_extractor = StateExtractor()
    action_space = ActionSpace()
    
    rollouts = {name: [] for name in agent_names}
    
    # Load checkpoint files
    checkpoint_files = sorted([
        f for f in os.listdir(checkpoint_path)
        if f.startswith("simulate-") and f.endswith(".json")
    ])
    
    for i in range(len(checkpoint_files) - 1):
        current_file = os.path.join(checkpoint_path, checkpoint_files[i])
        next_file = os.path.join(checkpoint_path, checkpoint_files[i + 1])
        
        with open(current_file, 'r') as f:
            current_data = json.load(f)
        
        with open(next_file, 'r') as f:
            next_data = json.load(f)
        
        # Extract states and actions from checkpoints
        for name in agent_names:
            if name not in current_data.get("agents", {}):
                continue
            
            current_agent = current_data["agents"][name]
            next_agent = next_data.get("agents", {}).get(name, {})
            
            # Reconstruct state from checkpoint data
            # (This requires loading agent from checkpoint, which is complex)
            # For simplicity, we'll skip detailed implementation here
            
            # Extract action from checkpoint
            action_info = current_agent.get("action", {})
            
            # Create transition
            transition = {
                "state": None,  # Would need to reconstruct
                "action": None,  # Would need to map from action_info
                "reward": 0.0,  # Would need to compute
                "next_state": None,  # Would need to reconstruct
                "done": False,
            }
            
            rollouts[name].append(transition)
    
    return rollouts

