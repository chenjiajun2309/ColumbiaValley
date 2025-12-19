# ColumbiaValley – Generative Agents on Campus

ColumbiaValley is an English-localized fork of the Generative Agents framework with a Columbia-themed campus, new agents, and a reworked replay UI (fixed top controls, bottom persona bar, adaptive zoom, etc.). This project integrates **Multi-Agent Proximal Policy Optimization (MAPPO)** for reinforcement learning-based agent behavior optimization, while maintaining LLM-based semantic reasoning and natural language interaction.

This README explains how to configure the environment, run simulations, train RL policies, and replay results.

---

## Table of Contents

- [1. Environment Setup](#1-environment-setup)
- [2. Run a Simulation](#2-run-a-simulation)
- [3. Reinforcement Learning Training](#3-reinforcement-learning-training)
- [4. Replay a Simulation](#4-replay-a-simulation)
- [5. Project Structure](#5-project-structure)
- [6. Configuration](#6-configuration)
- [7. Training Results and Visualization](#7-training-results-and-visualization)
- [8. Troubleshooting](#8-troubleshooting)
- [9. Project Highlights](#9-project-highlights)
- [10. References](#10-references)
- [11. License](#11-license)

---

## 1. Environment Setup

### 1.1 Clone the repository

```bash
git clone https://github.com/chenjiajun2309/ColumbiaValley.git
cd ColumbiaValley
```

### 1.2 Configure the language model backend

We use [Ollama](https://ollama.com/) by default to serve:
- Chat model: `qwen3:8b-q4_K_M`
- Embedding model: `bge-m3`

**Install Ollama:**
- Download from https://ollama.com/
- Pull required models:
  ```bash
  ollama pull qwen3:8b-q4_K_M
  ollama pull bge-m3:latest
  ```
- Start Ollama service:
  ```bash
  ollama serve
  ```
  Default port: `11434`

**Edit `generative_agents/data/config.json`:**

```json
{
  "provider": "ollama",
  "model": "qwen3:8b-q4_K_M",
  "base_url": "http://127.0.0.1:11434",
  "embedding_model": "bge-m3:latest"
}
```

If you prefer an OpenAI-compatible API, set `provider` to `"openai"` and supply `api_key`, `model`, and `base_url`.

> See `docs/ollama.md` for full instructions on installing and configuring Ollama for this project.

### 1.3 Install Python dependencies

Python 3.12 is recommended. Using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Key dependencies:**
- `torch>=2.0.0` - PyTorch for RL training
- `numpy>=1.24.0` - Numerical computations
- `Flask==3.1.1` - Replay server
- `openai==1.98.0` - LLM API client
- `tqdm>=4.65.0` - Progress bars
- `matplotlib>=3.7.0` - Visualization

---

## 2. Run a Simulation

All commands in this section are executed inside the `generative_agents` directory.

### 2.1 Basic simulation

```bash
cd generative_agents
python start.py --name columbia1 --start "20250213-09:30" --step 120 --stride 10
```

**Key arguments:**
- `--name`: simulation identifier (data stored in `results/checkpoints/<name>`)
- `--start`: simulation start time in `YYYYMMDD-HH:MM` format
- `--step`: number of steps to simulate
- `--stride`: minutes advanced per step (e.g., stride=10 → 09:00 → 09:10 → 09:20)
- `--resume`: resume an existing simulation

### 2.2 Additional options

**Verbosity control:**
```bash
--verbose debug    # Most detailed (default)
--verbose info     # Info level
--verbose warning  # Warnings only (faster)
--verbose error    # Errors only
```

**Logging to file:**
```bash
--log simulation.log  # Saves logs to results/checkpoints/<name>/simulation.log
```

**Parallel processing (faster for multiple agents):**
```bash
--parallel --max_workers 4  # Use 4 parallel workers
```

### 2.3 Output

After each run you'll see:
- Checkpoint JSON files in `results/checkpoints/<name>/`
- Per-agent memory data
- RL training data (if `--use_rl` is enabled, see Section 3)

---

## 3. Reinforcement Learning Training

ColumbiaValley integrates **MAPPO (Multi-Agent Proximal Policy Optimization)** to train agent policies using reinforcement learning, while agents still use LLMs for semantic reasoning and natural language generation.

### 3.1 Enable RL training

Add `--use_rl` flag to enable RL training:

```bash
cd generative_agents
python start.py \
    --name test-rl \
    --use_rl \
    --step 100 \
    --rl_train_interval 10
```

**Key RL parameters:**
- `--use_rl` or `--use-rl`: Enable RL training
- `--rl_train_interval`: Training frequency (train every N steps), default: `10`

### 3.2 How RL training works

1. **Data Collection**: During simulation, the `OnlineDataCollector` hooks into agent decision-making:
   - Extracts state features (persona, spatial, action, memory, social, schedule)
   - Records LLM-generated actions (mapped to discrete RL actions)
   - Computes multi-component rewards
   - Stores transitions `(s_t, a_t, r_t, s_{t+1})` in rollout buffers

2. **Training**: At fixed intervals (`rl_train_interval`):
   - Flushes rollout buffers
   - Computes GAE (Generalized Advantage Estimation) advantages
   - Updates policy and value networks using PPO clipped objective
   - Saves model checkpoints periodically

3. **Action Space**: 6 discrete actions
   - `CONTINUE (0)`: Continue current activity
   - `INITIATE_CHAT (1)`: Start conversation with nearby agent
   - `WAIT (2)`: Wait for another agent
   - `CHANGE_LOCATION (3)`: Change location
   - `REVISE_SCHEDULE (4)`: Modify schedule
   - `SKIP_REACTION (5)`: Skip reacting to events

4. **Reward Function**: Multi-component reward (weights sum to 100%):
   - Persona Alignment (25%): Keyword-based persona-action matching
   - Interaction Quality (35%): Conversation success and length
   - Relationship Growth (25%): Relationship score changes
   - Diversity (10%): Penalizes excessive repetition
   - Schedule Completion (15%): Rewards matching planned actions

### 3.3 Training examples

**Quick test (short training):**
```bash
python start.py \
    --name test-rl-quick \
    --use_rl \
    --step 50 \
    --rl_train_interval 5 \
    --verbose info
```

**Full training run:**
```bash
python start.py \
    --name columbia1-rl \
    --use_rl \
    --rl_train_interval 10 \
    --step 500 \
    --stride 15 \
    --verbose info \
    --log training.log
```

**Training with parallel processing:**
```bash
python start.py \
    --name columbia1-rl-parallel \
    --use_rl \
    --parallel \
    --max_workers 4 \
    --step 200 \
    --rl_train_interval 10
```

### 3.4 Model checkpoints

Trained models are saved to:
```
results/checkpoints/<name>/rl_models/
```

Checkpoints are saved every `10 × rl_train_interval` steps by default.

> See `docs/rl_usage_guide.md` and `docs/mappo_implementation_guide.md` for detailed RL documentation.

---

## 4. Replay a Simulation

### 4.1 Generate replay data

```bash
cd generative_agents
python compress.py --name columbia1
```

This produces:
- `results/compressed/<name>/movement.json` (frame-by-frame movement + actions)
- `results/compressed/<name>/simulation.md` (text timeline of agent states/conversations)

### 4.2 Start the replay server

> **Important:** Use the Flask server (`python replay.py`). Do **not** use `http.server`, otherwise the Jinja2 templates will not render and you'll lose the UI.

```bash
cd generative_agents
python replay.py
```

When you see `Running on http://127.0.0.1:5000/`, open the browser:

```
http://127.0.0.1:5000/?name=columbia1
```

**Controls:**
- Arrow keys: Pan the camera
- Mouse wheel: Zoom in/out

**URL parameters:**
- `name`: simulation name (required)
- `step`: starting frame (default 0)
- `speed`: replay speed (0–5, default 2)
- `zoom`: initial zoom ratio (default 0.8)

**Example:**
```
http://127.0.0.1:5000/?name=columbia1&step=0&speed=2&zoom=0.6
```

### 4.3 Replay troubleshooting

| Issue | Cause | Fix |
| --- | --- | --- |
| UI buttons missing | Serving via `http.server` | Launch with `python replay.py` |
| Agents missing / white squares | Static assets not found or atlas mismatch | Check Network tab for 404s; verify atlas JSONs (`generate_atlas_json.py`) |
| Agents don't move | Movement data unchanged | Inspect `movement.json`; rerun simulation with more steps |
| Console `replayData is undefined` | Jinja template not rendered | Use Flask server; ensure `<script id="replay-data">` block is intact |
| Replay finishes instantly | Already at last frame | Button shows `[Replay finished]`; rerun simulation or lower start step |
| UI still in Chinese | Mixed-language templates | Ensure `frontend/templates/index.html` and `main_script.html` from this repo are deployed |

---

## 5. Project Structure

```
ColumbiaValley/
├── generative_agents/
│   ├── modules/
│   │   ├── agent.py              # Agent class with RL integration
│   │   ├── game.py               # Game environment with RL trainer
│   │   ├── maze.py               # Campus map
│   │   ├── memory/               # Memory systems (spatial, schedule, etc.)
│   │   ├── model/                # LLM model interface
│   │   ├── rl/                   # RL components
│   │   │   ├── mappo/            # MAPPO implementation
│   │   │   │   ├── trainer.py   # MAPPO trainer
│   │   │   │   └── network.py   # Policy and value networks
│   │   │   ├── data_collector.py # Online data collection
│   │   │   ├── reward_function.py # Multi-component reward
│   │   │   ├── state_extractor.py # State feature extraction
│   │   │   └── action_space.py  # Action space definition
│   │   └── utils/                # Utility functions
│   ├── data/
│   │   ├── config.json           # LLM configuration
│   │   └── prompts/              # LLM prompts
│   ├── frontend/                 # Replay UI
│   │   ├── templates/            # Jinja2 templates
│   │   └── static/               # Static assets
│   ├── start.py                  # Main simulation script
│   ├── compress.py               # Generate replay data
│   ├── replay.py                 # Replay server
│   └── compare_baseline.py       # Compare RL vs baseline
├── docs/                         # Documentation
│   ├── ollama.md                 # Ollama setup guide
│   ├── rl_usage_guide.md         # RL usage guide
│   ├── mappo_implementation_guide.md
│   └── ...                       # Additional docs
├── results/
│   ├── checkpoints/              # Simulation checkpoints
│   │   └── <name>/
│   │       ├── rl_models/        # Trained RL models
│   │       ├── rl_metrics.json  # Training metrics
│   │       └── rl_visualizations/ # Training plots
│   └── compressed/               # Replay data
└── requirements.txt              # Python dependencies
```

---

## 6. Configuration

### 6.1 LLM configuration (`generative_agents/data/config.json`)

```json
{
  "provider": "ollama",
  "model": "qwen3:8b-q4_K_M",
  "base_url": "http://127.0.0.1:11434",
  "embedding_model": "bge-m3:latest",
  "timeout": 300,
  "max_retries": 3
}
```

### 6.2 RL training configuration

RL hyperparameters can be configured in `Game.__init__()` or via command-line arguments. Default values:

- Discount factor (`gamma`): `0.99`
- GAE parameter (`lambda`): `0.95`
- Clip parameter (`epsilon`): `0.2`
- Value loss coefficient: `0.5`
- Entropy coefficient: `0.01`
- Max gradient norm: `0.5`
- Training epochs per update: `10`
- Batch size: `64`
- Learning rate (policy & value): `3e-4`

### 6.3 Reward function weights

Configured in `RewardFunction.__init__()`:
- Persona Alignment: `25%`
- Interaction Quality: `35%`
- Relationship Growth: `25%`
- Diversity: `10%`
- Schedule Completion: `15%`

---

## 7. Training Results and Visualization

### 7.1 View training metrics

After running RL training, metrics are saved to:
```
results/checkpoints/<name>/rl_metrics.json
```

Visualizations are automatically generated in:
```
results/checkpoints/<name>/rl_visualizations/
```

**Generated plots:**
- `policy_ratios.png` - Policy update ratios (should stay within [0.8, 1.2])
- `kl_divergence.png` - KL divergence between consecutive policies
- `training_losses.png` - Policy loss, value loss, and entropy
- `gradient_norms.png` - Gradient norms (should stay below max threshold)
- `reward_components.png` - Reward component breakdown
- `reward_trends.png` - Overall reward trends
- `reward_distribution.png` - Reward distribution histograms
- `action_distribution.png` - Action selection distribution
- `learning_curves.png` - Cumulative returns over episodes
- `value_estimates.png` - Value function estimates

### 7.2 Compare RL vs Baseline

To compare RL-trained agents with baseline (non-RL) agents:

```bash
cd generative_agents
python compare_baseline.py \
    --rl_checkpoint results/checkpoints/test-rl \
    --baseline_checkpoint results/checkpoints/test-baseline
```

This generates comparison plots showing the effectiveness of RL training.

### 7.3 Analyze training data

You can load and analyze training metrics in Python:

```python
import json

with open("results/checkpoints/test-rl/rl_metrics.json", "r") as f:
    metrics = json.load(f)

# Check episode returns
if "episode_returns" in metrics:
    for agent, returns in metrics["episode_returns"].items():
        print(f"{agent}: {len(returns)} episodes")
        print(f"  Mean return: {sum(returns)/len(returns):.3f}")
```

> See `docs/rl_training_analysis.md` for detailed analysis examples.

---

## 8. Troubleshooting

### 8.1 Common issues

**Issue: Ollama connection error**
- **Solution**: Ensure Ollama is running (`ollama serve`) and check `base_url` in `config.json`

**Issue: CUDA out of memory (RL training)**
- **Solution**: Reduce batch size or number of agents, or use CPU mode (automatically falls back if CUDA unavailable)

**Issue: Training metrics not generated**
- **Solution**: Ensure `--use_rl` flag is set and training actually occurred (check logs for "Training on X transitions")

**Issue: Agents not learning (all actions are CONTINUE)**
- **Solution**: 
  - Increase training duration (`--step 500+`)
  - Adjust reward function weights
  - Check if reward components are providing meaningful signals

**Issue: Replay UI not loading**
- **Solution**: 
  - Use Flask server (`python replay.py`), not `http.server`
  - Check browser console for errors
  - Verify `movement.json` exists in `results/compressed/<name>/`

### 8.2 Performance optimization

**Speed up simulation:**
- Use `--parallel --max_workers 4` for multi-agent simulations
- Set `--verbose warning` to reduce logging overhead
- Use smaller models or faster quantization (e.g., `qwen3:8b-q4_K_M`)

**Speed up RL training:**
- Increase `--rl_train_interval` (train less frequently)
- Reduce batch size (in trainer code)
- Use GPU if available (automatic)

**Reduce memory usage:**
- Reduce number of agents
- Decrease `--step` count
- Clear old checkpoints periodically

---

## 9. Project Highlights

- **12 Columbia-themed agents** with English bios, schedules, and locations
- **Phaser-based map** with adaptive zoom, top toolbar, and fixed persona bar
- **Ollama/OpenAI-compatible LLM layer** with fallback logic and improved error handling
- **MAPPO reinforcement learning** for policy optimization while maintaining LLM-based reasoning
- **Multi-component reward function** balancing persona alignment, interaction quality, relationship growth, diversity, and schedule completion
- **Online data collection** integrated into agent decision loop
- **Comprehensive training visualization** with automatic metric plotting
- **Replay UI improvements**:
  - Top controls remain aligned despite zoom changes
  - Conversation box scales with viewport
  - Bottom persona bar stays centered and scrollable
  - Agent sprites enlarge (2×) while preserving texture slicing

---

## 10. References

### Papers
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) - Original generative agents paper
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - PPO algorithm
- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) - MAPPO algorithm

### Original Projects
- [joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents) - Original Stanford implementation
- [Archermmt/wounderland](https://github.com/Archermmt/wounderland) - Chinese localization fork

### Documentation
- See `docs/` directory for detailed guides:
  - `docs/ollama.md` - Ollama setup and configuration
  - `docs/rl_usage_guide.md` - RL training guide
  - `docs/mappo_implementation_guide.md` - MAPPO implementation details
  - `docs/command_line_arguments.md` - Complete command-line reference

---

## 11. License

This project is licensed under the Apache License 2.0. See `LICENSE` file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## Acknowledgments

We would like to express our gratitude to:
- The original Generative Agents team for the foundational framework
- The open-source community for valuable feedback and contributions
- Columbia University for providing the academic environment for this research

---

**For questions or issues, please open an issue on GitHub or refer to the documentation in the `docs/` directory.**
