# Verbose级别说明

## 问题

使用 `--verbose warning` 后看不到agents的活动信息，这是因为：

### 日志级别

```
DEBUG < INFO < WARN < ERROR < CRITICAL
```

### 显示规则

- `--verbose debug`：显示所有日志（DEBUG, INFO, WARN, ERROR）
- `--verbose info`：显示 INFO, WARN, ERROR（不显示DEBUG）
- `--verbose warning`：只显示 WARN, ERROR（不显示INFO和DEBUG）
- `--verbose error`：只显示 ERROR

### Agents活动信息

Agents的活动信息（如思考过程、决策等）都是用 `logger.info()` 输出的，所以：

- ✅ `--verbose debug` 或 `--verbose info`：可以看到
- ❌ `--verbose warning`：看不到（被过滤了）

## 解决方案

### 方案1：使用info级别（推荐）

```bash
python start.py \
    --name test-rl \
    --use_rl \
    --step 20 \
    --rl_train_interval 5 \
    --parallel \
    --max_workers 4 \
    --verbose info \  # 改为info
    --stride 15
```

**效果**：
- ✅ 显示agents活动
- ✅ 显示重要信息
- ❌ 不显示详细调试信息（减少噪音）

### 方案2：使用debug级别（最详细）

```bash
--verbose debug
```

**效果**：
- ✅ 显示所有信息（包括详细调试）
- ⚠️ 输出很多，可能影响性能

### 方案3：保存到日志文件

```bash
python start.py \
    --name test-rl \
    --use_rl \
    --step 20 \
    --rl_train_interval 5 \
    --parallel \
    --max_workers 4 \
    --verbose info \
    --log simulation.log \  # 保存到文件
    --stride 15
```

**日志文件位置**：`results/checkpoints/test-rl/simulation.log`

### 方案4：混合使用（终端warning，文件info）

可以修改代码支持不同级别的文件日志，或者：
- 终端：`--verbose warning`（减少输出）
- 文件：自动保存所有info级别日志

## 推荐配置

### 快速测试（只看进度条）
```bash
--verbose warning
```

### 正常使用（看agents活动）
```bash
--verbose info
```

### 调试问题（看所有细节）
```bash
--verbose debug
```

### 长时间运行（保存日志）
```bash
--verbose info --log simulation.log
```

## 查看日志文件

如果使用了 `--log` 参数：

```bash
# 查看日志
tail -f results/checkpoints/test-rl/simulation.log

# 或查看最后100行
tail -n 100 results/checkpoints/test-rl/simulation.log
```

## 总结

- **`--verbose warning`** = 只显示警告和错误，不显示agents活动
- **`--verbose info`** = 显示agents活动和重要信息（推荐）
- **`--verbose debug`** = 显示所有信息（最详细）

要看到agents活动，请使用 `--verbose info` 或 `--verbose debug`。

