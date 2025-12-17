# 命令行参数完整说明

## 所有可用参数

### 基本参数

#### `--name` (必需)
**说明**：Simulation名称，用于创建checkpoint文件夹

**用法**：
```bash
--name test-rl
```

**默认值**：如果未提供，会提示输入

---

#### `--start`
**说明**：Simulation的开始时间（游戏内时间）

**用法**：
```bash
--start 20240213-09:30
```

**默认值**：`20240213-09:30`

**格式**：`YYYYMMDD-HH:MM`

---

#### `--step`
**说明**：运行多少个simulation步骤

**用法**：
```bash
--step 20
```

**默认值**：`10`

---

#### `--stride`
**说明**：每个step推进多少分钟（游戏内时间）

**用法**：
```bash
--stride 15  # 每个step推进15分钟
```

**默认值**：`10`（每个step推进10分钟）

**示例**：
- `--step 20 --stride 15` = 运行20步，每步15分钟 = 总共5小时游戏时间

---

#### `--resume`
**说明**：从之前的checkpoint恢复运行

**用法**：
```bash
--resume --name columbia1
```

**说明**：会从`results/checkpoints/{name}`中加载最新的checkpoint继续运行

---

### 日志参数

#### `--verbose`
**说明**：日志详细程度

**用法**：
```bash
--verbose debug    # 最详细（默认）
--verbose info     # 信息级别
--verbose warning  # 警告级别（最快）
--verbose error    # 仅错误
```

**默认值**：`debug`

**建议**：
- 调试时使用：`debug`
- 正常运行时使用：`info`或`warning`（更快）

---

#### `--log`
**说明**：将日志保存到文件

**用法**：
```bash
--log simulation.log
```

**默认值**：空（输出到控制台）

**保存位置**：`results/checkpoints/{name}/{log_file}`

---

### RL训练参数

#### `--use_rl` 或 `--use-rl`
**说明**：启用RL训练

**用法**：
```bash
--use_rl
# 或
--use-rl
```

**说明**：启用后会：
- 初始化OnlineDataCollector
- 初始化MAPPOTrainer
- 在simulation过程中收集数据并训练

---

#### `--rl_train_interval` 或 `--rl-train-interval`
**说明**：RL训练间隔（每N步训练一次）

**用法**：
```bash
--rl_train_interval 5   # 每5步训练一次
--rl_train_interval 10  # 每10步训练一次（默认）
```

**默认值**：`10`

**建议**：
- 快速测试：`5`
- 正常训练：`10`
- 节省计算：`20`

---

### 并行处理参数

#### `--parallel`
**说明**：启用并行agent处理（加速simulation）

**用法**：
```bash
--parallel
```

**效果**：2-10倍加速（取决于agent数量）

**注意**：需要多个agents才能看到效果

---

#### `--max_workers`
**说明**：并行处理的最大worker数量

**用法**：
```bash
--max_workers 4   # 使用4个并行worker（默认）
--max_workers 8   # 使用8个并行worker
```

**默认值**：`4`（或agent数量，取较小值）

**建议**：
- CPU核心数：`2-4`
- 如果有GPU：`4-8`
- 不要超过agent数量

---

## 完整命令示例

### 示例1：基本运行
```bash
python start.py --name test --step 20
```

### 示例2：RL训练
```bash
python start.py \
    --name test-rl \
    --use_rl \
    --step 20 \
    --rl_train_interval 5
```

### 示例3：快速运行（并行+减少日志）
```bash
python start.py \
    --name test-fast \
    --parallel \
    --max_workers 4 \
    --step 20 \
    --verbose warning
```

### 示例4：完整配置（RL+并行+日志）
```bash
python start.py \
    --name columbia1-rl \
    --use_rl \
    --rl_train_interval 10 \
    --parallel \
    --max_workers 4 \
    --step 100 \
    --stride 15 \
    --verbose info \
    --log training.log
```

### 示例5：恢复运行
```bash
python start.py \
    --name columbia1 \
    --resume \
    --step 50
```

### 示例6：长时间运行（保存日志）
```bash
python start.py \
    --name long-run \
    --use_rl \
    --parallel \
    --step 500 \
    --stride 10 \
    --verbose info \
    --log long_run.log
```

## 参数组合建议

### 快速测试
```bash
python start.py \
    --name test \
    --step 5 \
    --verbose warning \
    --parallel
```

### 正常训练
```bash
python start.py \
    --name training \
    --use_rl \
    --rl_train_interval 10 \
    --step 100 \
    --verbose info
```

### 生产运行
```bash
python start.py \
    --name production \
    --use_rl \
    --parallel \
    --max_workers 4 \
    --step 1000 \
    --stride 15 \
    --verbose warning \
    --log production.log
```

## 查看帮助

```bash
python start.py --help
```

会显示所有参数的说明。

## 参数优先级

1. **命令行参数** > **config.json** > **默认值**
2. 如果同时指定，命令行参数会覆盖config.json中的设置

## 注意事项

1. **`--name`** 是必需的（或会提示输入）
2. **`--resume`** 需要checkpoint文件夹存在
3. **`--parallel`** 需要多个agents才能看到效果
4. **`--use_rl`** 需要安装torch等依赖
5. **`--max_workers`** 不要超过CPU核心数太多

## 环境变量

除了命令行参数，还可以设置环境变量：

```bash
# Ollama并行处理
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2

# 然后运行
python start.py --name test --parallel
```

