# Ollama Setup for ColumbiaValley

The ColumbiaValley agent stack uses an on-premise [Ollama](https://ollama.com/) runtime to serve both the chat model (`qwen3:8b-q4_K_M`) and the embedding model (`bge-m3`). Follow this guide to install Ollama, download the required models, and expose the API so `generative_agents` can connect to `http://localhost:11434`.

---

## 1. Install Ollama

1. Download the latest release from [https://ollama.com/](https://ollama.com/).  
2. Run the installer with the default options.  
3. Verify the CLI is available:

```bash
ollama --version
```

> **Hardware guidance**  
> * Apple Silicon (M-series) with ≥16 GB RAM or Windows + RTX 30/40 GPUs with ≥12 GB VRAM are recommended for `qwen3:8b-q4_K_M`.  
> * If you have ≥24 GB VRAM you may experiment with higher precision models, but our config expects the `q4_K_M` quantization.

---

## 2. Pull Project Models

The ColumbiaValley repo expects two models. Download them once:

```bash
ollama pull qwen3:8b-q4_K_M
ollama pull bge-m3:latest
```

Use `ollama list` to confirm they are available.

---

## 3. Start the Service

You can launch Ollama via the GUI or the CLI:

```bash
ollama serve
```

By default the server listens on `http://127.0.0.1:11434`. Our `generative_agents/data/config.json` points to this host/port, so no additional wiring is necessary as long as the service is running before you start `python generative_agents/start.py`.

---

## 4. Allow API Requests

To avoid HTTP 403 errors you must expose the API bindings.

### macOS

```bash
launchctl setenv OLLAMA_HOST "0.0.0.0"
launchctl setenv OLLAMA_ORIGINS "*"
```

### Windows

Add system environment variables (*System Properties → Advanced → Environment Variables*):

```
OLLAMA_HOST=0.0.0.0
OLLAMA_ORIGINS=*
```

Restart the Ollama service after adding the variables.

---

## 5. Useful Environment Tweaks

| Purpose | Variable | Example |
| ------- | -------- | ------- |
Bind to a different address/port | `OLLAMA_HOST` | `OLLAMA_HOST=0.0.0.0:11500` |
Serve on all interfaces | `OLLAMA_HOST` | `OLLAMA_HOST=0.0.0.0` |
Move model cache off the system drive | `OLLAMA_MODELS` | `OLLAMA_MODELS=D:\OllamaModels` |
Keep models in RAM longer (default 5 min) | `OLLAMA_KEEP_ALIVE` | `OLLAMA_KEEP_ALIVE=2h` |
Allow multiple concurrent requests | `OLLAMA_NUM_PARALLEL` | `OLLAMA_NUM_PARALLEL=2` |
Permit multiple models loaded at once | `OLLAMA_MAX_LOADED_MODELS` | `OLLAMA_MAX_LOADED_MODELS=2` |

Define these before `ollama serve` starts (e.g., in your shell profile or system env vars).

---

## 6. Verify Connectivity

1. Ensure `ollama serve` is running.  
2. From the repo root, call:

```bash
curl http://localhost:11434/api/tags
```

You should see a JSON list of installed models.  
3. Launch the simulation (`python generative_agents/start.py`) and confirm the logs show `OllamaLLMModel` responses without authentication errors.

If you still see 401/403 responses, double-check the `OLLAMA_HOST`/`OLLAMA_ORIGINS` values and restart the service.

---
