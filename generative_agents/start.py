import os
import copy
import json
import argparse
import datetime
import time

from dotenv import load_dotenv, find_dotenv

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: create a dummy tqdm class
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        def set_description(self, desc):
            pass
        def set_postfix(self, **kwargs):
            pass

from modules.game import create_game, get_game
from modules import utils

personas = [
    "Ava_Lee", 
    "Benjamin_Carter", "Daniel_Kim",
    #"Evelyn_Park",
    # "Grace_Chen", "Jason_Wright", "Liam_OConnor", "Marta_Lopez",
    # "Noah_Patel", "Priya_Nair", "Rosa_Martinez", "Sophia_Rossi",
]


class SimulateServer:
    def __init__(self, name, static_root, checkpoints_folder, config, start_step=0, verbose="info", log_file=""):
        self.name = name
        self.static_root = static_root
        self.checkpoints_folder = checkpoints_folder

        # Historical checkpoint data (for resume)
        self.config = config

        os.makedirs(checkpoints_folder, exist_ok=True)

        # Load historical conversation data (for resume)
        self.conversation_log = f"{checkpoints_folder}/conversation.json"
        if os.path.exists(self.conversation_log):
            with open(self.conversation_log, "r", encoding="utf-8") as f:
                conversation = json.load(f)
        else:
            conversation = {}

        if len(log_file) > 0:
            self.logger = utils.create_file_logger(f"{checkpoints_folder}/{log_file}", verbose)
        else:
            self.logger = utils.create_io_logger(verbose)

        # Create game
        game = create_game(name, static_root, config, conversation, logger=self.logger)
        game.reset_game()

        self.game = get_game()
        self.tile_size = self.game.maze.tile_size
        self.agent_status = {}
        if "agent_base" in config:
            agent_base = config["agent_base"]
        else:
            agent_base = {}
        for agent_name, agent in config["agents"].items():
            agent_config = copy.deepcopy(agent_base)
            agent_config.update(self.load_static(agent["config_path"]))
            self.agent_status[agent_name] = {
                "coord": agent_config["coord"],
                "path": [],
            }
        self.think_interval = max(
            a.think_config["interval"] for a in self.game.agents.values()
        )
        self.start_step = start_step
        
        # RL training configuration
        # Note: Trainer is already initialized in Game.__init__() if use_rl=True
        self.rl_train_interval = config.get("rl_train_interval", 10)  # Train every N steps
        self.rl_enabled = config.get("use_rl", False) and self.game.rl_trainer is not None
        
        if self.rl_enabled:
            self.logger.info(f"RL training enabled (interval: {self.rl_train_interval} steps)")
        elif config.get("use_rl", False):
            self.logger.warning("RL is enabled in config but trainer is not available")

    def simulate(self, step, stride=0):
        timer = utils.get_timer()
        total_steps = step
        start_time = time.time()
        
        # Create progress bar
        with tqdm(
            total=total_steps,
            initial=self.start_step,
            desc="Simulation",
            unit="step",
            disable=not TQDM_AVAILABLE,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) as pbar:
            for i in range(self.start_step, self.start_step + step):
                step_start_time = time.time()
                current_step = i + 1 - self.start_step
                
                title = "Simulate Step[{}/{}, time: {}]".format(current_step, total_steps, timer.get_date())
                self.logger.info("\n" + utils.split_line(title, "="))
                
                # Update progress bar description
                sim_time_str = timer.get_date("%Y%m%d-%H:%M")
                pbar.set_description(f"Step {current_step}/{total_steps} @ {sim_time_str}")
                
                # Each agent thinks (this will automatically collect RL data if enabled)
                agent_start_time = time.time()
                
                # Update training step for metrics recorder (so rewards are recorded with correct step)
                if self.rl_enabled and self.game.rl_collector and self.game.rl_collector.metrics_recorder:
                    # Use current simulation step (i+1) for reward recording
                    self.game.rl_collector.metrics_recorder.training_step = i + 1
                
                for name, status in self.agent_status.items():
                    plan = self.game.agent_think(name, status)["plan"]
                    agent = self.game.get_agent(name)
                    if name not in self.config["agents"]:
                        self.config["agents"][name] = {}
                    self.config["agents"][name].update(agent.to_dict())
                    if plan.get("path"):
                        status["coord"], status["path"] = plan["path"][-1], []
                    self.config["agents"][name].update(
                        # {"coord": status["coord"], "path": plan["path"]}
                        {"coord": status["coord"]}
                    )
                
                agent_time = time.time() - agent_start_time

                # RL Training: Collect data and train periodically
                rl_train_time = 0
                if self.rl_enabled and (i + 1) % self.rl_train_interval == 0:
                    rl_start_time = time.time()
                    # Use Game's rl_train_step method (which uses the trainer initialized in Game)
                    self.game.rl_train_step(i + 1)
                    rl_train_time = time.time() - rl_start_time

                sim_time = timer.get_date("%Y%m%d-%H:%M")
                self.config.update(
                    {
                        "time": sim_time,
                        "step": i + 1,
                    }
                )
                # Save Agent activity data
                with open(f"{self.checkpoints_folder}/simulate-{sim_time.replace(':', '')}.json", "w", encoding="utf-8") as f:
                    f.write(json.dumps(self.config, indent=2, ensure_ascii=False))
                # Save conversation data
                with open(f"{self.checkpoints_folder}/conversation.json", "w", encoding="utf-8") as f:
                    f.write(json.dumps(self.game.conversation, indent=2, ensure_ascii=False))

                if stride > 0:
                    timer.forward(stride)
                
                # Update progress bar with timing information
                step_time = time.time() - step_start_time
                elapsed_time = time.time() - start_time
                avg_time_per_step = elapsed_time / current_step if current_step > 0 else 0
                remaining_steps = total_steps - current_step
                estimated_remaining = avg_time_per_step * remaining_steps
                
                postfix_info = {
                    "step_time": f"{step_time:.2f}s",
                    "agent_time": f"{agent_time:.2f}s",
                }
                if rl_train_time > 0:
                    postfix_info["rl_train"] = f"{rl_train_time:.2f}s"
                if remaining_steps > 0:
                    postfix_info["ETA"] = f"{estimated_remaining/60:.1f}m"
                
                pbar.set_postfix(**postfix_info)
                pbar.update(1)
        
        # Print summary
        total_time = time.time() - start_time
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Simulation completed!")
        self.logger.info(f"Total steps: {total_steps}")
        self.logger.info(f"Total time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
        self.logger.info(f"Average time per step: {total_time/total_steps:.2f} seconds")
        if self.rl_enabled:
            self.logger.info(f"RL training: Enabled (interval: {self.rl_train_interval} steps)")
        self.logger.info(f"{'='*60}\n")
        
        # Generate RL metrics visualizations (for both RL and baseline)
        # Even if use_rl=False, we still generate visualizations if metrics were recorded
        self.logger.info(f"Checking RL visualization conditions...")
        self.logger.info(f"  rl_enabled: {self.rl_enabled}")
        self.logger.info(f"  rl_collector exists: {self.game.rl_collector is not None}")
        if self.game.rl_collector:
            self.logger.info(f"  metrics_recorder exists: {self.game.rl_collector.metrics_recorder is not None}")
        
        # Generate visualizations if collector exists (for both RL and baseline)
        if self.game.rl_collector and self.game.rl_collector.metrics_recorder:
            try:
                self.logger.info("Generating RL metrics visualizations...")
                from modules.rl.visualizer import RLMetricsVisualizer
                
                # Check if metrics recorder has any data
                summary = self.game.rl_collector.metrics_recorder.get_summary()
                self.logger.info(f"Metrics summary: {summary}")
                
                # Debug: Check reward history directly
                reward_history = self.game.rl_collector.metrics_recorder.reward_history
                self.logger.info(f"Reward history keys: {list(reward_history.keys())}")
                for agent_name, history in reward_history.items():
                    self.logger.info(f"  {agent_name}: {len(history)} rewards recorded")
                
                if not summary or not summary.get("agents"):
                    self.logger.warning("No RL metrics data found. Skipping visualization generation.")
                    self.logger.warning("This might mean:")
                    self.logger.warning("  1. No rewards were recorded during simulation")
                    self.logger.warning("  2. RL collector was not properly initialized")
                    self.logger.warning("  3. Agents did not take any actions that generated rewards")
                    self.logger.warning("  4. end_collection was called but action_dict was None")
                    
                    # Try to save metrics anyway (might have some data)
                    metrics_file = self.game.rl_collector.metrics_recorder.save_metrics()
                    if metrics_file:
                        self.logger.info(f"RL metrics saved to: {metrics_file} (may be empty)")
                    return
                else:
                    # Ensure checkpoints_folder is set
                    if not self.game.rl_collector.metrics_recorder.checkpoints_folder:
                        self.game.rl_collector.metrics_recorder.checkpoints_folder = self.checkpoints_folder
                        self.logger.info(f"Set checkpoints_folder to: {self.checkpoints_folder}")
                    
                    # Save metrics
                    metrics_file = self.game.rl_collector.metrics_recorder.save_metrics()
                    if metrics_file:
                        self.logger.info(f"RL metrics saved to: {metrics_file}")
                    else:
                        self.logger.warning("Failed to save RL metrics file (save_metrics returned None)")
                        return
                    
                    # Generate visualizations
                    viz_dir = os.path.join(self.checkpoints_folder, "rl_visualizations")
                    os.makedirs(viz_dir, exist_ok=True)
                    
                    visualizer = RLMetricsVisualizer(metrics_file, viz_dir)
                    visualizer.generate_all_visualizations()
                    
                    self.logger.info(f"RL visualizations saved to: {viz_dir}")
                    self.logger.info("  - reward_trends.png")
                    self.logger.info("  - reward_components.png")
                    self.logger.info("  - reward_distribution.png")
                    self.logger.info("  - action_distribution.png")
                    if self.rl_enabled:
                        self.logger.info("  - learning_curves.png")
                        self.logger.info("  - training_losses.png")
                    self.logger.info("  - rl_summary.txt")
                
            except Exception as e:
                self.logger.warning(f"Failed to generate RL visualizations: {e}")
                import traceback
                self.logger.warning(traceback.format_exc())
        else:
            if self.rl_enabled:
                self.logger.warning("RL is enabled but metrics recorder is not available")
                if not self.game.rl_collector:
                    self.logger.warning("  - RL collector is None")
                elif not self.game.rl_collector.metrics_recorder:
                    self.logger.warning("  - Metrics recorder is None")
            else:
                # Baseline mode: still try to save metrics if collector exists
                if self.game.rl_collector and self.game.rl_collector.metrics_recorder:
                    try:
                        # Ensure checkpoints_folder is set
                        if not self.game.rl_collector.metrics_recorder.checkpoints_folder:
                            self.game.rl_collector.metrics_recorder.checkpoints_folder = self.checkpoints_folder
                        
                        # Save metrics
                        metrics_file = self.game.rl_collector.metrics_recorder.save_metrics()
                        if metrics_file:
                            self.logger.info(f"Baseline metrics saved to: {metrics_file}")
                            
                            # Generate visualizations for baseline
                            from modules.rl.visualizer import RLMetricsVisualizer
                            viz_dir = os.path.join(self.checkpoints_folder, "rl_visualizations")
                            os.makedirs(viz_dir, exist_ok=True)
                            visualizer = RLMetricsVisualizer(metrics_file, viz_dir)
                            visualizer.generate_all_visualizations()
                            self.logger.info(f"Baseline visualizations saved to: {viz_dir}")
                    except Exception as e:
                        self.logger.warning(f"Failed to save baseline metrics: {e}")
    

    def load_static(self, path):
        return utils.load_dict(os.path.join(self.static_root, path))


# Load configuration from checkpoint data for resume
def get_config_from_log(checkpoints_folder):
    files = sorted(os.listdir(checkpoints_folder))

    json_files = list()
    for file_name in files:
        if file_name.endswith(".json") and file_name != "conversation.json":
            json_files.append(os.path.join(checkpoints_folder, file_name))

    if len(json_files) < 1:
        return None

    with open(json_files[-1], "r", encoding="utf-8") as f:
        config = json.load(f)

    assets_root = os.path.join("assets", "village")

    start_time = datetime.datetime.strptime(config["time"], "%Y%m%d-%H:%M")
    start_time += datetime.timedelta(minutes=config["stride"])
    config["time"] = {"start": start_time.strftime("%Y%m%d-%H:%M")}
    agents = config["agents"]
    for a in agents:
        config["agents"][a]["config_path"] = os.path.join(assets_root, "agents", a.replace(" ", "_"), "agent.json")

    return config


# Create configuration for new game
def get_config(start_time="20240213-09:30", stride=15, agents=None):
    with open("data/config.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
        agent_config = json_data["agent"]

    assets_root = os.path.join("assets", "village")
    config = {
        "stride": stride,
        "time": {"start": start_time},
        "maze": {"path": os.path.join(assets_root, "maze.json")},
        "agent_base": agent_config,
        "agents": {},
    }
    for a in agents:
        config["agents"][a] = {
            "config_path": os.path.join(
                assets_root, "agents", a.replace(" ", "_"), "agent.json"
            ),
        }
    return config


load_dotenv(find_dotenv())

parser = argparse.ArgumentParser(description="console for village")
parser.add_argument("--name", type=str, default="", help="The simulation name")
parser.add_argument("--start", type=str, default="20240213-09:30", help="The starting time of the simulated ville")
parser.add_argument("--resume", action="store_true", help="Resume running the simulation")
parser.add_argument("--step", type=int, default=10, help="The simulate step")
parser.add_argument("--stride", type=int, default=10, help="The step stride in minute")
parser.add_argument("--verbose", type=str, default="debug", help="The verbose level")
parser.add_argument("--log", type=str, default="", help="Name of the log file")
parser.add_argument("--use_rl", "--use-rl", action="store_true", dest="use_rl", help="Enable RL training")
parser.add_argument("--rl_train_interval", "--rl-train-interval", type=int, default=10, dest="rl_train_interval", help="RL training interval (steps)")
args = parser.parse_args()


if __name__ == "__main__":
    checkpoints_path = "results/checkpoints"

    name = args.name
    if len(name) < 1:
        name = input("Please enter a simulation name (e.g. sim-test): ")

    resume = args.resume
    if resume:
        while not os.path.exists(f"{checkpoints_path}/{name}"):
            name = input(f"'{name}' doesn't exists, please re-enter the simulation name: ")
    else:
        while os.path.exists(f"{checkpoints_path}/{name}"):
            name = input(f"The name '{name}' already exists, please enter a new name: ")

    checkpoints_folder = f"{checkpoints_path}/{name}"

    start_time = args.start
    if resume:
        sim_config = get_config_from_log(checkpoints_folder)
        if sim_config is None:
            print("No checkpoint file found to resume running.")
            exit(0)
        start_step = sim_config["step"]
    else:
        sim_config = get_config(start_time, args.stride, personas)
        start_step = 0

    static_root = "frontend/static"
    
    # Add RL configuration if enabled
    if args.use_rl:
        if "rl" not in sim_config:
            sim_config["rl"] = {}
        sim_config["use_rl"] = True
        sim_config["rl_train_interval"] = args.rl_train_interval
        # Default RL config (can be overridden in config file)
        if "trainer" not in sim_config["rl"]:
            sim_config["rl"]["trainer"] = {
                "lr": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": 0.2,
                "rollout_length": 2048,
                "num_epochs": 10,
            }
        if "collector" not in sim_config["rl"]:
            sim_config["rl"]["collector"] = {
                "max_buffer_size": 1000,
            }

    server = SimulateServer(name, static_root, checkpoints_folder, sim_config, start_step, args.verbose, args.log)
    server.simulate(args.step, args.stride)
