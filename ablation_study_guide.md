# 消融实验指南

## 📊 消融实验配置说明

本项目支持对各个改进点进行消融实验，通过修改YAML配置文件中的`ablation`参数来控制各个模块的开启/关闭。

## 🔧 可配置的改进点

| 参数名                   | 说明               | 默认值  | 影响模块            |
| ------------------------ | ------------------ | ------- | ------------------- |
| `use_small_branch`       | 是否使用小目标分支 | `true`  | SmallObjectBranch   |
| `use_ghost`              | 是否使用Ghost模块  | `true`  | GhostModule         |
| `use_attention`          | 是否使用注意力机制 | `true`  | LocalAttention      |
| `use_pos_encoding`       | 是否使用位置编码   | `true`  | 相对位置编码        |
| `use_cross_scale_fusion` | 是否使用跨尺度融合 | `true`  | CrossScaleAttention |
| `use_teacher`            | 是否使用知识蒸馏   | `false` | 教师-学生蒸馏       |

## 📁 预配置的消融实验文件

### 1. `v11-small-full.yaml` - 完整模型

所有改进点都开启，这是性能最好的配置。

### 2. `v11-small-wo-ghost.yaml` - 无Ghost模块

测试Ghost模块的贡献，使用标准卷积替代。

### 3. `v11-small-wo-small-branch.yaml` - 无小目标分支

测试小目标分支的贡献，直接使用主分支特征。

### 4. `v11-small-wo-attention.yaml` - 无注意力机制

测试注意力机制的贡献，使用标准卷积替代。

### 5. `v11-small-wo-pos-encoding.yaml` - 无位置编码

测试位置编码的贡献，保留注意力机制但去掉位置编码。

### 6. `v11-small-wo-fusion.yaml` - 无跨尺度融合

测试跨尺度融合的贡献，直接使用主分支特征。

### 7. `v11-small-baseline.yaml` - 基线模型

关闭所有改进点，作为对比基线。

## 🚀 使用方法

### 方法1: 使用预配置的YAML文件

```python
from ultralytics import YOLO

# 测试完整模型
model_full = YOLO("v11-small-full.yaml")
model_full.train(data="v11-data.yaml", epochs=200, name="exp-full")

# 测试无Ghost模块
model_wo_ghost = YOLO("v11-small-wo-ghost.yaml")
model_wo_ghost.train(data="v11-data.yaml", epochs=200, name="exp-wo-ghost")

# 测试基线模型
model_baseline = YOLO("v11-small-baseline.yaml")
model_baseline.train(data="v11-data.yaml", epochs=200, name="exp-baseline")
```

### 方法2: 自定义配置

修改YAML文件中的`ablation`部分：

```yaml
# Parameters
nc: 80

# Ablation study configuration
ablation:
  use_teacher: false
  use_small_branch: true # 改为false关闭
  use_ghost: false # 改为false关闭
  use_attention: true
  use_pos_encoding: true
  use_cross_scale_fusion: true

backbone:
  - [custom, 1, YOLOv11SmallObjectDetector]

head:
  - [-1, 1, Detect, [nc]]
```

## 📈 推荐的消融实验顺序

1. **基线模型** (`v11-small-baseline.yaml`) - 建立性能基准
2. **+小目标分支** - 测试小目标分支的贡献
3. **+Ghost模块** - 测试Ghost模块的贡献
4. **+注意力机制** - 测试注意力机制的贡献
5. **+位置编码** - 测试位置编码的贡献
6. **+跨尺度融合** - 测试跨尺度融合的贡献
7. **完整模型** (`v11-small-full.yaml`) - 最终性能

## 📝 实验记录建议

建议记录以下指标进行对比：

- **mAP@0.5**: 平均精度（IoU=0.5）
- **mAP@0.5:0.95**: 平均精度（IoU=0.5-0.95）
- **参数量 (Params)**: 模型参数量
- **FLOPs**: 浮点运算次数
- **推理速度 (FPS)**: 每秒处理帧数
- **训练时间**: 每个epoch的训练时间

## 🔍 注意事项

1. **依赖关系**:
   - 关闭`use_small_branch`时，`use_cross_scale_fusion`会自动失效
   - 关闭`use_attention`时，`use_pos_encoding`会自动失效

2. **公平对比**:
   - 确保所有实验使用相同的数据集、训练参数和随机种子
   - 建议多次运行取平均值

3. **计算资源**:
   - 完整模型的计算量最大
   - 基线模型的计算量最小

## 💡 示例实验脚本

```python
from ultralytics import YOLO

# 消融实验配置列表
ablation_configs = [
    ("v11-small-baseline.yaml", "baseline"),
    ("v11-small-wo-ghost.yaml", "wo-ghost"),
    ("v11-small-wo-small-branch.yaml", "wo-small-branch"),
    ("v11-small-wo-attention.yaml", "wo-attention"),
    ("v11-small-wo-pos-encoding.yaml", "wo-pos-encoding"),
    ("v11-small-wo-fusion.yaml", "wo-fusion"),
    ("v11-small-full.yaml", "full"),
]

for config_file, exp_name in ablation_configs:
    print(f"\n{'=' * 50}")
    print(f"Training with {config_file} - {exp_name}")
    print(f"{'=' * 50}\n")

    model = YOLO(config_file)
    model.train(
        data="v11-data.yaml",
        epochs=200,
        batch=16,
        imgsz=640,
        optimizer="SGD",
        amp=True,
        project="runs/ablation",
        name=exp_name,
    )
```

## 📊 结果分析

训练完成后，对比各个配置的性能指标，可以得出：

1. **每个改进点的独立贡献**
2. **改进点之间的协同效应**
3. **性能与计算量的权衡**

---

**提示**: 建议先在小数据集上快速验证配置是否正确，再在完整数据集上进行正式实验。
