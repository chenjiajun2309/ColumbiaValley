# GPU配置说明

## 当前状态

**之前**：所有RL训练都在CPU上运行

**现在**：已添加GPU支持，会自动检测并使用GPU（如果可用）

## 自动检测

代码会自动检测GPU可用性：

1. **如果有GPU可用**：自动使用GPU（`cuda`）
2. **如果没有GPU**：自动使用CPU

启动时会显示：
```
🔧 Using device: cuda
   GPU: NVIDIA GeForce RTX 3090
   CUDA Version: 11.8
```

或

```
🔧 Using device: cpu
```

## 手动配置

### 在config.json中配置

```json
{
    "use_rl": true,
    "rl": {
        "trainer": {
            "device": "cuda",        // 强制使用GPU
            "device": "cpu",         // 强制使用CPU
            "device": "cuda:0",     // 使用特定GPU（多GPU环境）
            "device": null,          // 自动检测（默认）
            // ... 其他配置
        }
    }
}
```

### 设备选项

- `"cuda"` - 使用第一个可用GPU
- `"cpu"` - 强制使用CPU
- `"cuda:0"` - 使用GPU 0
- `"cuda:1"` - 使用GPU 1
- `null` 或不设置 - 自动检测（推荐）

## 性能对比

### CPU vs GPU

| 操作 | CPU时间 | GPU时间 | 加速比 |
|------|---------|---------|--------|
| 小批量训练 | ~100ms | ~10ms | 10x |
| 大批量训练 | ~5s | ~0.5s | 10x |
| 模型推理 | ~50ms | ~5ms | 10x |

*注：实际性能取决于模型大小、批量大小和硬件配置*

## 检查GPU状态

### 在Python中检查

```python
import torch

# 检查CUDA是否可用
print(f"CUDA available: {torch.cuda.is_available()}")

# 检查GPU数量
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
```

### 命令行检查

```bash
# 检查NVIDIA GPU
nvidia-smi

# 检查CUDA版本
nvcc --version
```

## 内存管理

### GPU内存

训练时会自动使用GPU内存。如果遇到内存不足：

1. **减小批量大小**：
```json
{
    "rl": {
        "trainer": {
            "batch_size": 32  // 从64减小到32
        }
    }
}
```

2. **使用CPU**（如果GPU内存不足）：
```json
{
    "rl": {
        "trainer": {
            "device": "cpu"
        }
    }
}
```

### 监控GPU使用

```bash
# 实时监控GPU使用
watch -n 1 nvidia-smi
```

## 多GPU支持

如果有多块GPU，可以指定使用哪一块：

```json
{
    "rl": {
        "trainer": {
            "device": "cuda:0"  // 使用第一块GPU
            "device": "cuda:1"  // 使用第二块GPU
        }
    }
}
```

## 故障排除

### 问题1：CUDA不可用

**错误**：
```
RuntimeError: CUDA error: no kernel image is available for execution
```

**解决**：
- 检查PyTorch是否支持CUDA：`python -c "import torch; print(torch.cuda.is_available())"`
- 重新安装支持CUDA的PyTorch：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### 问题2：GPU内存不足

**错误**：
```
RuntimeError: CUDA out of memory
```

**解决**：
- 减小`batch_size`
- 减小`rollout_length`
- 使用CPU：`"device": "cpu"`

### 问题3：模型加载错误

**错误**：
```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
```

**解决**：
- 确保模型和输入在同一设备上（代码已自动处理）

## 最佳实践

1. **自动检测**：不设置`device`，让代码自动选择
2. **监控内存**：训练时监控GPU内存使用
3. **批量大小**：根据GPU内存调整`batch_size`
4. **保存设备信息**：模型保存时会保存设备信息

## 代码实现

主要修改：

1. **设备检测**：`_get_device()`方法自动检测或使用配置
2. **模型移到GPU**：`.to(self.device)`将模型移到指定设备
3. **Tensor移到GPU**：`.to(self.device)`将所有tensor移到指定设备
4. **模型加载**：`map_location=self.device`确保模型加载到正确设备

## 验证

运行时会看到设备信息：

```
🔧 Using device: cuda
   GPU: NVIDIA GeForce RTX 3090
   CUDA Version: 11.8
```

如果没有GPU：

```
🔧 Using device: cpu
```

