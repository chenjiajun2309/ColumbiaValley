# ColumbiaValley – Generative Agents on Campus

ColumbiaValley is an English-localized fork of the Generative Agents framework with a Columbia-themed campus, new agents, and a reworked replay UI (fixed top controls, bottom persona bar, adaptive zoom, etc.). This README explains how to configure the environment, run simulations, and replay results.

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

Edit `generative_agents/data/config.json`:

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
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Run a Simulation

All commands in this section are executed inside the `generative_agents` directory.

```bash
cd generative_agents
python start.py --name columbia1 --start "20250213-09:30" --step 120 --stride 10
```

Key arguments:
- `--name`: simulation identifier (data stored in `results/checkpoints/<name>`)
- `--start`: simulation start time in `YYYYMMDD-HH:MM` format
- `--step`: number of steps to simulate
- `--stride`: minutes advanced per step (e.g., stride=10 → 09:00 → 09:10 → 09:20)
- `--resume`: resume an existing simulation

After each run you’ll see checkpoint JSON files plus per-agent memory data under `results/checkpoints/<name>/`.

---

## 3. Replay a Simulation

### 3.1 Generate replay data

```bash
cd generative_agents
python compress.py --name columbia1
```

This produces:
- `results/compressed/<name>/movement.json` (frame-by-frame movement + actions)
- `results/compressed/<name>/simulation.md` (text timeline of agent states/conversations)

### 3.2 Start the replay server

> **Important:** Use the Flask server (`python replay.py`). Do **not** use `http.server`, otherwise the Jinja2 templates will not render and you’ll lose the UI.

```bash
cd generative_agents
python replay.py
```

When you see `Running on http://127.0.0.1:5000/`, open the browser:

```
http://127.0.0.1:5000/?name=columbia1
```

Use arrow keys to pan the camera. Query parameters:
- `name`: simulation name (required)
- `step`: starting frame (default 0)
- `speed`: replay speed (0–5, default 2)
- `zoom`: initial zoom ratio (default 0.8)

### 3.3 Replay Troubleshooting

| Issue | Cause | Fix |
| --- | --- | --- |
UI buttons missing | Serving via `http.server` | Launch with `python replay.py` |
Agents missing / white squares | Static assets not found or atlas mismatch | Check Network tab for 404s; verify atlas JSONs (`generate_atlas_json.py`) |
Agents don’t move | Movement data unchanged | Inspect `movement.json`; rerun simulation with more steps |
Console `replayData is undefined` | Jinja template not rendered | Use Flask server; ensure `<script id="replay-data">` block is intact |
Replay finishes instantly | Already at last frame | Button shows `[Replay finished]`; rerun simulation or lower start step |
UI still in Chinese | Mixed-language templates | Ensure `frontend/templates/index.html` and `main_script.html` from this repo are deployed |

---

## 4. Project Highlights

- 12 Columbia-themed agents with English bios, schedules, and locations.
- Phaser-based map with adaptive zoom, top toolbar, and fixed persona bar.
- Ollama/OpenAI-compatible LLM layer with fallback logic, improved error handling.
- Replay UI improvements:  
  * top controls remain aligned despite zoom changes  
  * conversation box scales with viewport  
  * bottom persona bar stays centered and scrollable  
  * agent sprites enlarge (2×) while preserving texture slicing

See `docs/` for additional guides (e.g., `docs/ollama.md`).

---

## 5. References

- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)  
- Original projects:  
  * [joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents)  
  * [Archermmt/wounderland](https://github.com/Archermmt/wounderland)

---
