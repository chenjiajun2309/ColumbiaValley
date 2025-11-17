[简体中文](./README.md) | English

# Generative Agents Chinesized

## 1. Configure the environment

### 1.1 pull the source code:

```
git clone https://github.com/x-glacier/GenerativeAgentsCN.git
cd GenerativeAgentsCN
```

### 1.2 configure the large language model

Modify the configuration file `generative_agents/data/config.json`:
1. By default, [Ollama](https://ollama.com/) is used to load local quantization models and OpenAI compatible APIs are provided. We need to first pull the quantization model and ensure that `base_url` and `model` are consistent with the actual configuration of Ollama.
2. If you want to call other OpenAI compatible APIs, you need to change `provider` to `openai`, and modify `model`, `api_key` and `base_url` to the correct values.

### 1.3 install python dependencies

Use a virtual environment, e.g. with anaconda3:

```
conda create -n generative_agents_cn python=3.12
conda activate generative_agents_cn
```

Install dependencies:

```
pip install -r requirements.txt
```

## 2. Start a simulation

```
cd generative_agents
python start.py --name sim-test --start "20240213-09:30" --step 10 --stride 10
```

arguments:
- `name` - the name of the simulation
- `start` - the starting time of the simulated ville
- `resume` - resume running the simulation
- `step` - how many steps to simulate
- `stride` - how many minutes to forward after each step, e.g. 9:00->9:10->9:20 if stride=10

## 3. Replay a simulation

### 3.1 generate replay data

```
python compress.py --name <simulation-name>
```

After running, the replay data file `movement.json` will be generated in the `results/compressed/<simulation-name>` folder. At the same time, `simulation.md` will be generated to present the status and conversation of each agent in a timeline.

### 3.2 start the replay server

**⚠️ Important: You must use Flask server, NOT `http.server`!**

The replay interface depends on Flask + Jinja2 template rendering. If you use a static server, UI controls (Run/Pause/Show Chat buttons) will not display.

**Correct way to start:**

```bash
cd generative_agents
python replay.py
```

The terminal should show:
```
* Running on http://127.0.0.1:5000/  (Press CTRL+C to quit)
```

**❌ Wrong way (DO NOT use):**
```bash
python -m http.server 5173 -d generative_agents/frontend/static
```
This will cause templates to fail rendering and all UI will disappear.

**Visit the replay page:**

Open in browser: `http://127.0.0.1:5000/?name=<simulation-name>`

*Use arrow keys to move the camera*

**URL parameters:**
- `name` - The name of the simulation (required)
- `step` - Starting step of replay, 0 means from the first frame, default is 0
- `speed` - Replay speed (0-5), 0 is slowest, 5 is fastest, default is 2
- `zoom` - Zoom ratio, default is 0.8

**Example:**
```
http://127.0.0.1:5000/?name=sim-test&step=0&speed=2&zoom=0.6
```

### 3.3 Troubleshooting Replay Issues

#### Issue 1: UI buttons (Run/Pause/Show Chat) not displaying

**Symptoms:** No control buttons visible after page loads, only map is shown.

**Cause:** Using `http.server` instead of Flask, causing Jinja2 templates (`{% ... %}` and `{{ ... }}`) to not render.

**Solution:**
1. Confirm you're using `python replay.py` to start Flask server
2. Check terminal output shows `Running on http://127.0.0.1:5000/`
3. If you see `Serving HTTP on :: port ...`, you're using `http.server` incorrectly - stop it and use Flask instead

#### Issue 2: Agents not displaying or showing as white blocks

**Possible causes:**

1. **Incorrect resource paths:** Check browser Developer Tools Network panel for 404 errors
   - Correct path: `/static/assets/village/agents/<Name>/texture.png`
   - If you see 404, check if `frontend/static/assets/village/` directory exists

2. **Atlas frame name mismatch:** Run in browser Console:
   ```javascript
   game.scene.scenes[0].textures.get('Ava_Lee').getFrameNames()
   ```
   Confirm frame names are in format like `down`, `left-walk.000`, etc.

3. **Agents blocked by foreground layer:** Ensure agent sprite depth is set correctly:
   ```javascript
   new_sprite.setDepth(1.5);  // Should be greater than foreground layer depth
   ```

#### Issue 3: Agents not moving or animations not playing

**Possible causes:**

1. **Coordinates unchanged in data:** Check `results/compressed/<name>/movement.json`, confirm `movement` coordinates change across different steps
   - If all steps have same coordinates, simulation ran too briefly and agents didn't have time to move
   - Solution: Re-run simulation with increased `--step` parameter (e.g., `--step 120`)

2. **Data format mismatch:** Confirm agent names in `movement.json` use underscores (e.g., `Ava_Lee`), not spaces (e.g., `Ava Lee`)

3. **Replay finished:** Check if button shows `[Replay finished]`, if so, replay has reached the last frame

#### Issue 4: Console error `replayData is undefined`

**Cause:** Template variables not rendered correctly, JSON data block is empty.

**Solution:**
1. Confirm you're using Flask server (not `http.server`)
2. Check `<script id="replay-data">` tag in `frontend/templates/index.html` is correct
3. Refresh page (Cmd+Shift+R for hard refresh)

#### Issue 5: Incomplete translation to English

**Symptoms:** Some UI elements still display in Chinese.

**Solution:** Check if these files have been updated to English:
- `frontend/templates/index.html` - Button text and hints
- `frontend/templates/main_script.html` - Button labels and conversation text

## 4. Reference

### 4.1 paper

[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)

### 4.2 gitHub repository

[Generative Agents](https://github.com/joonspk-research/generative_agents)

[wounderland](https://github.com/Archermmt/wounderland)
