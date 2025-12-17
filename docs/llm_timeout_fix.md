# LLM超时和解析错误修复

## 问题描述

运行过程中遇到两个问题：

1. **超时错误**：
   ```
   HTTPConnectionPool(host='127.0.0.1', port=11434): Read timed out. (read timeout=60)
   ```

2. **解析错误**：
   ```
   parse_llm_output: Patterns tried: [...]
   Failed to match llm output
   ```

## 已修复的问题

### 1. 超时时间增加

**之前**：固定60秒超时
**现在**：默认300秒（5分钟），可配置

**修改**：
- 默认超时从60秒增加到300秒
- 支持在config.json中配置`timeout`参数

### 2. 解析错误处理改进

**之前**：使用`assert`，失败时程序崩溃
**现在**：更优雅的错误处理

**改进**：
- 添加`failsafe`参数，可以返回默认值
- 改进错误信息，显示更多调试信息
- 如果`ignore_empty=True`，返回None而不是崩溃

## 配置方法

### 在config.json中设置超时

```json
{
    "agent": {
        "think": {
            "llm": {
                "provider": "ollama",
                "model": "qwen3:8b-q4_K_M",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "...",
                "timeout": 300  // 超时时间（秒），默认300
            }
        }
    }
}
```

### 超时时间建议

- **小模型（<7B）**：120-180秒
- **中等模型（7B-13B）**：180-300秒
- **大模型（>13B）**：300-600秒
- **复杂prompt**：根据实际情况增加

## 使用parse_llm_output的改进

### 之前（会崩溃）

```python
result = parse_llm_output(response, patterns)
# 如果匹配失败，程序崩溃
```

### 现在（更安全）

```python
# 方式1：使用failsafe
result = parse_llm_output(response, patterns, failsafe="default_value")

# 方式2：使用ignore_empty
result = parse_llm_output(response, patterns, ignore_empty=True)
if result is None:
    # 处理无匹配的情况
    result = "default_value"
```

## 错误处理流程

### LLM调用错误处理

代码已经实现了重试机制：

```python
for _ in range(retry):  # 默认重试10次
    try:
        response = self._completion(prompt, **kwargs)
        # 成功，退出循环
        break
    except Exception as e:
        print(f"LLMModel.completion() caused an error: {e}")
        time.sleep(5)  # 等待5秒后重试
        continue
```

### 超时错误

如果遇到超时：
1. 自动重试（最多10次）
2. 每次重试前等待5秒
3. 如果所有重试都失败，使用`failsafe`值（如果有）

## 调试建议

### 1. 检查Ollama服务

```bash
# 检查Ollama是否运行
curl http://127.0.0.1:11434/api/tags

# 检查模型是否加载
ollama list
```

### 2. 监控Ollama日志

```bash
# 查看Ollama日志
tail -f ~/.ollama/logs/server.log
```

### 3. 测试简单请求

```python
import requests

response = requests.post(
    "http://127.0.0.1:11434/v1/chat/completions",
    json={
        "model": "qwen3:8b-q4_K_M",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.5
    },
    timeout=300
)
print(response.json())
```

## 常见问题

### Q1: 仍然超时怎么办？

**A**: 
1. 增加`timeout`值（如600秒）
2. 检查Ollama服务是否正常
3. 检查模型是否太大，考虑使用更小的模型
4. 检查网络连接

### Q2: 解析仍然失败怎么办？

**A**:
1. 检查LLM输出格式是否符合预期
2. 调整正则表达式模式
3. 使用`failsafe`参数提供默认值
4. 使用`ignore_empty=True`避免崩溃

### Q3: 如何查看完整的LLM响应？

**A**: 
错误信息中已经包含了响应预览（前500字符）。如果需要完整响应，可以：
1. 在代码中添加日志
2. 检查`agent._llm.meta_responses`（包含所有响应）

## 性能优化建议

1. **使用更小的模型**：如果不需要高质量输出，使用量化模型
2. **减少prompt长度**：精简prompt可以减少处理时间
3. **调整temperature**：较低temperature通常更快
4. **批量处理**：如果可能，批量处理请求

## 总结

- ✅ 超时时间从60秒增加到300秒（可配置）
- ✅ 改进了错误处理，避免程序崩溃
- ✅ 添加了failsafe机制
- ✅ 改进了错误信息，便于调试

现在运行应该更稳定了！

