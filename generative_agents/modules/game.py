"""generative_agents.game"""

import os
import copy

from modules.utils import GenerativeAgentsMap, GenerativeAgentsKey
from modules import utils
from .maze import Maze
from .agent import Agent


class Game:
    """The Game"""

    def __init__(self, name, static_root, config, conversation, logger=None):
        self.name = name
        self.static_root = static_root
        self.record_iterval = config.get("record_iterval", 30)
        self.logger = logger or utils.IOLogger()
        self.maze = Maze(self.load_static(config["maze"]["path"]), self.logger)
        self.conversation = conversation
        self.agents = {}
        if "agent_base" in config:
            agent_base = config["agent_base"]
        else:
            agent_base = {}
        storage_root = os.path.join(f"results/checkpoints/{name}", "storage")
        if not os.path.isdir(storage_root):
            os.makedirs(storage_root)
        for name, agent in config["agents"].items():
            agent_config = utils.update_dict(
                copy.deepcopy(agent_base), self.load_static(agent["config_path"])
            )
            agent_config = utils.update_dict(agent_config, agent)

            agent_config["storage_root"] = os.path.join(storage_root, name)
            self.agents[name] = Agent(agent_config, self.maze, self.conversation, self.logger)
        
        # Store RL config for later use
        self._rl_config = config.get("rl", {})
        
        # Initialize RL Data Collector for metrics recording
        # Even if use_rl=False, we still initialize collector to record metrics for baseline comparison
        checkpoints_folder = os.path.join(f"results/checkpoints/{name}")
        use_rl = config.get("use_rl", False)
        
        try:
            from modules.rl.data_collector import OnlineDataCollector
            rl_config = config.get("rl", {})
            collector_config = rl_config.get("collector", {})
            collector_config["checkpoints_folder"] = checkpoints_folder
            # Pass metrics config to collector
            collector_config["metrics"] = rl_config.get("metrics", {})
            collector_config["metrics"]["checkpoints_folder"] = checkpoints_folder
            # Mark if this is metrics-only mode (no training)
            collector_config["metrics_only"] = not use_rl
            self.rl_collector = OnlineDataCollector(collector_config)
            if use_rl:
                self.logger.info("RL Data Collector initialized (with training)")
            else:
                self.logger.info("RL Data Collector initialized (metrics only, no training)")
        except Exception as e:
            self.logger.warning(f"Failed to initialize RL Data Collector: {e}")
            import traceback
            self.logger.warning(traceback.format_exc())
            self.rl_collector = None
        
        # Initialize MAPPO Trainer if RL is enabled
        self.rl_trainer = None
        if use_rl:
            try:
                from modules.rl.mappo import MAPPOTrainer
                trainer_config = rl_config.get("trainer", {})
                # Get metrics_recorder from collector
                metrics_recorder = self.rl_collector.metrics_recorder if self.rl_collector else None
                self.rl_trainer = MAPPOTrainer(
                    config=trainer_config,
                    metrics_recorder=metrics_recorder
                )
                self.logger.info("MAPPO Trainer initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize MAPPO Trainer: {e}")
                import traceback
                self.logger.warning(traceback.format_exc())
                self.rl_trainer = None

    def get_agent(self, name):
        return self.agents[name]

    def agent_think(self, name, status):
        agent = self.get_agent(name)
        plan = agent.think(status, self.agents)
        info = {
            "currently": agent.scratch.currently,
            "associate": agent.associate.abstract(),
            "concepts": {c.node_id: c.abstract() for c in agent.concepts},
            "chats": [
                {"name": "self" if n == agent.name else n, "chat": c}
                for n, c in agent.chats
            ],
            "action": agent.action.abstract(),
            "schedule": agent.schedule.abstract(),
            "address": agent.get_tile().get_address(as_list=False),
        }
        if (
            utils.get_timer().daily_duration() - agent.last_record
        ) > self.record_iterval:
            info["record"] = True
            agent.last_record = utils.get_timer().daily_duration()
        else:
            info["record"] = False
        if agent.llm_available():
            info["llm"] = agent._llm.get_summary()
        title = "{}.summary @ {}".format(
            name, utils.get_timer().get_date("%Y%m%d-%H:%M:%S")
        )
        self.logger.info("\n{}\n{}\n".format(utils.split_line(title), agent))
        return {"plan": plan, "info": info}
    
    def rl_train_step(self, step: int, save_models_interval: int = 10):
        """
        Perform one RL training step
        
        This should be called at training intervals (e.g., every rl_train_interval steps)
        
        Args:
            step: Current simulation step
            save_models_interval: Save models every N training steps (default: 10)
        """
        if not self.rl_trainer or not self.rl_collector:
            return
        
        # Flush buffers to get collected rollouts
        rollouts = self.rl_collector.flush_all_buffers()
        
        if not rollouts or not any(len(r) > 0 for r in rollouts.values()):
            self.logger.info(f"Step {step}: No rollouts to train on")
            return
        
        # Train on collected data
        total_transitions = sum(len(r) for r in rollouts.values())
        self.logger.info(f"Step {step}: Training on {total_transitions} transitions")
        self.rl_trainer.train_step(rollouts, step=step)
        
        # Save models periodically
        rl_config = getattr(self, '_rl_config', {})
        rl_train_interval = rl_config.get("rl_train_interval", 10)
        training_step_count = step // rl_train_interval
        
        if training_step_count > 0 and training_step_count % save_models_interval == 0:
            checkpoints_folder = os.path.join(f"results/checkpoints/{self.name}")
            save_dir = os.path.join(checkpoints_folder, "rl_models", f"step_{step}")
            self.rl_trainer.save_models(save_dir)
            self.logger.info(f"Models saved to {save_dir}")

    def load_static(self, path):
        return utils.load_dict(os.path.join(self.static_root, path))

    def reset_game(self):
        for a_name, agent in self.agents.items():
            agent.reset()
            title = "{}.reset".format(a_name)
            self.logger.info("\n{}\n{}\n".format(utils.split_line(title), agent))


def create_game(name, static_root, config, conversation, logger=None):
    """Create the game"""

    utils.set_timer(**config.get("time", {}))
    GenerativeAgentsMap.set(GenerativeAgentsKey.GAME, Game(name, static_root, config, conversation, logger=logger))
    return GenerativeAgentsMap.get(GenerativeAgentsKey.GAME)


def get_game():
    """Get the gloabl game"""

    return GenerativeAgentsMap.get(GenerativeAgentsKey.GAME)
