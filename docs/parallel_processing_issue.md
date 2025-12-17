# 并行处理问题诊断

## 问题：并行处理没有加速

### 可能的原因

1. **Ollama不支持并发请求**
   - Ollama默认可能只处理一个请求
   - 需要设置环境变量启用并发

2. **LLM实例共享问题**
   - 每个agent可能共享同一个LLM实例
   - 需要检查是否有锁

3. **GIL（全局解释器锁）**
   - Python的GIL可能限制真正的并行
   - 但I/O操作（如HTTP请求）应该可以并行

## 解决方案

### 1. 设置Ollama并发

```bash
# 设置Ollama允许并发请求
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2

# 重启Ollama服务
# 然后运行
python start.py --name test --parallel --max_workers 4
```

### 2. 检查Ollama配置

编辑Ollama配置（如果存在）：
```bash
# 检查Ollama配置
cat ~/.ollama/config.json

# 或设置环境变量
export OLLAMA_NUM_PARALLEL=4
```

### 3. 验证并行是否工作

添加诊断信息来验证并行是否真正工作。

### 4. 使用多进程代替多线程

如果Ollama不支持并发，可以考虑使用多进程（但需要处理共享状态）。

## 诊断步骤

1. **检查Ollama是否支持并发**：
   ```bash
   # 测试并发请求
   curl http://127.0.0.1:11434/v1/chat/completions &
   curl http://127.0.0.1:11434/v1/chat/completions &
   # 看是否同时处理
   ```

2. **检查并行执行时间**：
   - 添加时间戳日志
   - 查看agents是否真正并行执行

3. **检查Ollama日志**：
   ```bash
   tail -f ~/.ollama/logs/server.log
   ```

## 临时解决方案

如果并行不工作，可以：
1. 减少agents数量
2. 使用更小的模型
3. 减少LLM调用次数

