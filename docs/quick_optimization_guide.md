# 快速优化指南

## 立即可以做的优化（5分钟）

### 1. 启用并行处理 ⭐⭐⭐⭐⭐

**效果**：2-10倍加速

```bash
python start.py --name test --parallel --max_workers 4 --step 20
```

### 2. 使用更小的模型 ⭐⭐⭐⭐

修改`data/config.json`：

```json
{
    "agent": {
        "think": {
            "llm": {
                "model": "qwen2.5:3b-q4_K_M"  // 从8b改为3b
            }
        },
        "chat_iter": 2,  // 从4改为2
        "think": {
            "poignancy_max": 300  // 从150改为300
        }
    }
}
```

### 3. 设置Ollama环境变量

```bash
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2

# 然后运行
python start.py --name test --parallel --step 20
```

## 完整优化命令

```bash
# 设置环境变量
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2

# 运行（使用并行+快速配置）
python start.py \
    --name test-fast \
    --parallel \
    --max_workers 4 \
    --step 20 \
    --verbose warning
```

## 预期加速效果

| 优化 | 单独加速 | 组合加速 |
|------|---------|---------|
| 并行处理 | 2-10x | - |
| 小模型(3b) | 2-5x | - |
| 减少chat_iter | 1.5-2x | - |
| **全部组合** | - | **5-20x** |

## 模型选择

### 最快（推荐用于测试）
```json
"model": "qwen2.5:1.5b-q4_K_M"
```

### 平衡（推荐用于生产）
```json
"model": "qwen2.5:3b-q4_K_M"
```

### 质量（当前）
```json
"model": "qwen3:8b-q4_K_M"
```

## 检查Ollama是否使用GPU

```bash
# 运行模型时检查
nvidia-smi

# 应该看到Ollama进程使用GPU
```

## 性能监控

运行时会显示：
- 每步耗时
- Agent处理时间
- 预计剩余时间

观察这些指标来评估优化效果。

