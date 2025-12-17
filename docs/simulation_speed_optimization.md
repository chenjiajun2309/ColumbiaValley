# Simulation速度优化指南

## 问题分析

**是的，速度主要取决于Ollama的2个模型！**

### 主要瓶颈

1. **LLM调用（主要瓶颈）**：
   - 每个agent每个step会进行**10-30次**LLM调用
   - 所有调用都是**串行**的（一个接一个）
   - 每个调用需要**2-10秒**（取决于模型大小）

2. **Embedding模型调用**：
   - 用于记忆检索
   - 相对较快，但也会累积

3. **GPU使用**：
   - GPU主要用于RL训练，不影响LLM调用速度
   - LLM调用是CPU/内存密集型

### 每个Agent的LLM调用次数（每个step）

```
make_schedule(): 3-5次
make_plan(): 0-2次  
reflect(): 2-10次（如果poignancy足够）
_chat_with(): 8-16次（chat_iter=4，每轮2次）
_determine_action(): 2-4次
```

**总计：每个agent每个step约15-40次LLM调用**

如果有12个agents，每个step就是**180-480次LLM调用**！

## 优化方案

### 方案1：并行化Agent处理 ⭐⭐⭐⭐⭐

**效果**：2-10倍加速（取决于agent数量）

**实现**：使用多线程/多进程并行处理agents

```python
# 在start.py的simulate方法中
from concurrent.futures import ThreadPoolExecutor, as_completed

def simulate(self, step, stride=0):
    # ... 原有代码 ...
    
    # 并行处理agents
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(self.game.agent_think, name, status): name
            for name, status in self.agent_status.items()
        }
        
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                plan = result["plan"]
                # ... 处理结果 ...
            except Exception as e:
                self.logger.error(f"Agent {name} failed: {e}")
```

### 方案2：减少LLM调用次数 ⭐⭐⭐⭐

#### 2.1 减少chat_iter

```json
{
    "agent": {
        "chat_iter": 2  // 从4减少到2，减少50%的聊天LLM调用
    }
}
```

#### 2.2 增加poignancy_max（减少reflect频率）

```json
{
    "agent": {
        "think": {
            "poignancy_max": 300  // 从150增加到300，减少reflect频率
        }
    }
}
```

#### 2.3 跳过某些非关键LLM调用

可以添加配置跳过某些调用：
- 跳过emoji生成（已注释）
- 跳过某些描述生成
- 使用缓存的结果

### 方案3：使用更小的模型 ⭐⭐⭐⭐

**效果**：2-5倍加速

```json
{
    "agent": {
        "think": {
            "llm": {
                "model": "qwen2.5:3b-q4_K_M"  // 从8b减少到3b
            }
        }
    }
}
```

**模型选择建议**：
- **最快**：`qwen2.5:1.5b-q4_K_M` 或 `qwen2.5:3b-q4_K_M`
- **平衡**：`qwen2.5:7b-q4_K_M`（当前使用8b）
- **质量**：`qwen3:8b-q4_K_M`（当前）

### 方案4：优化Ollama配置 ⭐⭐⭐

#### 4.1 使用更快的量化版本

```bash
# 使用Q4量化（更快）
ollama pull qwen2.5:7b-q4_K_M

# 或Q3量化（最快，但质量略降）
ollama pull qwen2.5:7b-q3_K_M
```

#### 4.2 增加Ollama并发数

```bash
# 设置Ollama环境变量
export OLLAMA_NUM_PARALLEL=4  # 允许4个并发请求
export OLLAMA_MAX_LOADED_MODELS=2  # 预加载2个模型
```

#### 4.3 使用GPU加速Ollama

```bash
# 确保Ollama使用GPU
ollama run qwen3:8b-q4_K_M
# 检查GPU使用
nvidia-smi
```

### 方案5：批量处理LLM请求 ⭐⭐⭐

**实现**：收集多个请求，批量发送

```python
# 需要修改LLMModel类支持批量请求
class OllamaLLMModel(LLMModel):
    def batch_completion(self, prompts, temperature=0.5):
        # 批量处理多个prompts
        # 需要Ollama支持批量API
        pass
```

### 方案6：缓存LLM响应 ⭐⭐

**实现**：缓存相似的prompt响应

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_completion(prompt_hash, prompt):
    return self._llm._completion(prompt)
```

### 方案7：减少日志输出 ⭐⭐

**效果**：轻微加速，减少I/O开销

```python
# 在start.py中
parser.add_argument("--verbose", type=str, default="warning")  # 从debug改为warning
```

## 推荐配置（快速模式）

### config.json

```json
{
    "agent": {
        "think": {
            "llm": {
                "provider": "ollama",
                "model": "qwen2.5:3b-q4_K_M",  // 更小的模型
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "...",
                "timeout": 180  // 减少超时
            },
            "poignancy_max": 300  // 减少reflect频率
        },
        "chat_iter": 2  // 减少聊天轮数
    }
}
```

### 环境变量

```bash
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2
```

## 性能对比

| 优化方案 | 加速比 | 质量损失 | 实现难度 |
|---------|--------|---------|---------|
| 并行化 | 2-10x | 无 | 中等 |
| 小模型 | 2-5x | 轻微 | 简单 |
| 减少chat_iter | 1.5-2x | 轻微 | 简单 |
| 增加poignancy_max | 1.2-1.5x | 无 | 简单 |
| 批量处理 | 1.5-3x | 无 | 困难 |
| 缓存 | 1.2-2x | 无 | 中等 |

## 实施优先级

1. **立即实施**（简单，效果好）：
   - ✅ 使用更小的模型（3b或7b）
   - ✅ 减少chat_iter到2
   - ✅ 增加poignancy_max到300

2. **短期实施**（需要修改代码）：
   - ✅ 并行化agent处理
   - ✅ 减少日志输出

3. **长期优化**（需要较大改动）：
   - ⚠️ 批量处理LLM请求
   - ⚠️ 实现响应缓存

## 快速测试

### 测试当前速度

```bash
time python start.py --name test-speed --step 5 --verbose warning
```

### 测试优化后速度

```bash
# 修改config.json后
time python start.py --name test-optimized --step 5 --verbose warning
```

## 监控LLM调用

在代码中添加统计：

```python
# 在Agent.completion()中
self._llm_call_count = getattr(self, '_llm_call_count', 0) + 1
if self._llm_call_count % 10 == 0:
    print(f"{self.name}: {self._llm_call_count} LLM calls")
```

## 总结

**主要瓶颈**：LLM调用（串行、频繁）

**最快优化**：
1. 并行化agent处理（2-10x加速）
2. 使用更小的模型（2-5x加速）
3. 减少chat_iter（1.5-2x加速）

**组合使用**：可以获得**5-20倍**的整体加速！

