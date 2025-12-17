# LLM解析问题说明

## 问题1：为什么会有这些Pattern显示？

### Pattern的作用

这些**Pattern（正则表达式）**不是为了让Ollama输出，而是为了**从Ollama的输出中提取结构化数据**。

### 工作流程

```
1. 代码发送Prompt给Ollama
   ↓
2. Ollama返回自然语言响应
   ↓
3. 代码使用Pattern（正则表达式）解析响应
   ↓
4. 提取结构化数据（如时间、活动等）
```

### Pattern定义位置

Pattern在代码中定义，用于匹配LLM输出：

**示例1：`schedule_daily` (scratch.py:186-193)**
```python
patterns = [
    r"\[(\d{1,2}:\d{2})\] " + self.name + r"(.*)。",
    r"\[(\d{1,2}:\d{2})\] " + self.name + r"(.*)",
    r"\[(\d{1,2}:\d{2})\] " + r"(.*)。",
    r"\[(\d{1,2}:\d{2})\] " + r"(.*)",
]
```

**示例2：`schedule_decompose` (scratch.py:223-230)**
```python
patterns = [
    r"\d{1,2}\) .*\*plans?\* (.*)[\(]+time[: ]+(\d{1,2})[, ]+remaining[: ]+\d*[\)]",
    r"\d{1,2}\) .*\*plans?\* (.*)[\(]+duration[: ]+(\d{1,2})[, ]+remaining[: ]+\d*[\)]",
    r"\d{1,2}\) .*\*plans?\* (.*)[\(]+(\d{1,2})[\)]",
    # ...
]
```

### 为什么显示这些Pattern？

当LLM输出**不匹配**任何Pattern时，代码会打印：
- `parse_llm_output: Failed to match. Response preview: ...`
- `parse_llm_output: Patterns tried: [...]`

这是**调试信息**，帮助开发者：
1. 了解LLM实际返回了什么
2. 知道哪些Pattern被尝试了
3. 诊断为什么解析失败

### 错误信息位置

在 `llm_model.py:330-331`：
```python
print(f"parse_llm_output: Failed to match. Response preview: {response[:500]}...")
print(f"parse_llm_output: Patterns tried: {patterns}")
```

---

## 问题2：多次调用Ollama导致输出问题？

### 问题分析

是的，**多次调用可能导致问题**，原因：

1. **上下文混乱**：
   - 每次调用都是独立的
   - LLM可能"解释"而不是"输出格式"
   - 没有记忆之前的格式要求

2. **Prompt不够严格**：
   - LLM倾向于"解释"和"帮助"
   - 而不是直接输出格式化内容

3. **重试机制**：
   - 代码有重试（最多10次）
   - 但每次重试都是同样的prompt
   - 如果prompt有问题，重试也没用

### 当前的重试机制

在 `llm_model.py:24-55`：
```python
def completion(self, prompt, retry=10, ...):
    for _ in range(retry):
        try:
            meta_response = self._completion(prompt, **kwargs).strip()
            # ... 处理响应
            if response is not None:
                break
        except Exception as e:
            print(f"LLMModel.completion() caused an error: {e}")
            time.sleep(5)
            continue
```

**问题**：重试时使用**相同的prompt**，如果prompt不够明确，重试也没用。

---

## 解决方案

### 方案1：改进Prompt（推荐）

在prompt中**更明确地要求格式**：

**当前prompt (schedule_daily.txt)**：
```
Please refer to the above character information and schedule breakdown to generate an hourly plan (24-hour format), only fill in the <activity> content, do not skip any time point.
The hourly plan must follow the format below:
"""
${hourly_schedule}
"""
```

**改进建议**：
```
IMPORTANT: You must output ONLY the schedule in the exact format below. Do NOT add explanations, comments, or additional text.

Required format (copy this format exactly):
"""
${hourly_schedule}
"""

Example:
[6:00] Wake up
[7:00] Eat breakfast
[8:00] Review lesson plans
...

Output ONLY the schedule lines, nothing else.
```

### 方案2：增强解析逻辑

已经实现了（在scratch.py中）：
- 支持多种格式（markdown、代码块等）
- 使用failsafe机制
- 更灵活的模式匹配

### 方案3：改进重试机制

在重试时**增强prompt**：

```python
def completion(self, prompt, retry=10, ...):
    for attempt in range(retry):
        try:
            # 在重试时添加更严格的格式要求
            if attempt > 0:
                prompt = prompt + "\n\nIMPORTANT: Output ONLY the formatted schedule. No explanations."
            
            meta_response = self._completion(prompt, **kwargs).strip()
            # ...
```

### 方案4：使用更小的模型或调整参数

- 使用更小的模型（如3b）可能更"听话"
- 调整temperature（降低随机性）
- 使用system prompt强调格式要求

---

## 当前代码的改进

### 已实现的改进

1. **多格式支持** (scratch.py:186-248)：
   - 原始格式：`[9:00] activity`
   - Markdown格式：`**9:00** activity`
   - 代码块格式
   - 自然语言格式

2. **Failsafe机制**：
   - 如果解析失败，使用默认值
   - 避免程序崩溃

3. **错误处理**：
   - 打印调试信息
   - 使用ignore_empty参数

### 建议的进一步改进

1. **增强Prompt**：
   - 在prompt中明确要求"只输出格式，不要解释"
   - 提供更清晰的示例

2. **后处理**：
   - 如果LLM返回解释性文本，尝试提取其中的格式部分
   - 使用更智能的文本提取

3. **减少调用**：
   - 缓存结果
   - 合并多个请求

---

## 总结

### 问题1答案

- **Pattern是代码中定义的**，用于解析LLM输出
- **不是为了让Ollama输出**，而是为了从输出中提取数据
- **显示Pattern是调试信息**，帮助诊断解析失败

### 问题2答案

- **是的，多次调用可能导致问题**
- **主要原因**：LLM倾向于"解释"而不是"输出格式"
- **解决方案**：改进prompt、增强解析、调整模型参数

### 建议

1. **短期**：增强prompt，明确要求格式
2. **中期**：改进解析逻辑，支持更多格式
3. **长期**：考虑使用更结构化的输出方式（如JSON）

