# YOLOv11 小目标检测改进模型

## 📋 项目简介

本项目基于 Ultralytics YOLOv11，针对航拍小目标检测场景进行了全面优化。模型集成了多个前沿研究成果，包括 Ghost 模块、小目标专用分支、Swin 注意力机制、跨尺度融合、知识蒸馏等先进技术，显著提升了小目标检测的精度和效率。

## ✨ 核心改进特性

### 1. **GhostModule - 轻量级特征提取**
- **来源**: FBRT-YOLO (Yao Xiao et al., AAAI 2025)
- **功能**: 通过 Ghost 操作实现高效特征提取，减少参数量和计算量
- **优势**: 在保持性能的同时显著降低模型复杂度
- **参考**: GhostNet 原始论文 (arXiv:1911.11907)

### 2. **SmallObjectBranch - 小目标专用分支**
- **来源**: I-YOLOv11n (Caiping Xi et al., Sensors 2025)
- **功能**: 为小目标检测设计的高分辨率特征路径
- **优势**: 增强 P2 层特征，强化小目标细节信息
- **参考**: [I-YOLOv11n 论文](https://www.mdpi.com/1424-8220/25/15/4857)

### 3. **双分支融合结构**
- **来源**: MASF-YOLO (Liugang Lu et al., arXiv 2025)
- **功能**: 同时提取主分支和小目标路径特征，增强空间细节
- **优势**: 通过多尺度特征融合提升小目标检测能力
- **参考**: [MASF-YOLO 论文](https://arxiv.org/abs/2504.18136)

### 4. **CrossScaleAttention - 跨尺度注意力融合**
- **来源**: CF-YOLO (Chengcheng Wang et al., Sci Reports 2025)
- **功能**: 多尺度特征融合，增强空间细节互补
- **优势**: 通过注意力机制实现跨尺度特征的有效融合
- **参考**: [CF-YOLO 论文](https://www.nature.com/articles/s41598-025-16741-w)

### 5. **LocalAttention + 相对位置编码**
- **来源**: Swin Transformer (Liu et al., ICCV 2021)
- **功能**: 使用局部注意力 + 位置感知，模拟 Swin 的窗口注意力机制
- **优势**: 增强模型对空间关系的建模能力，提升小目标定位精度
- **参考**: [Swin Transformer 论文](https://arxiv.org/abs/2103.14030)

### 6. **特征 MSE 损失 + KL 散度损失**
- **来源**: DistillDet (Zheng et al., ICCV 2022)
- **功能**: 教师-学生小目标知识迁移，增强特征与输出
- **优势**: 通过知识蒸馏提升小目标检测性能
- **参考**: [DistillDet 论文](https://arxiv.org/abs/2203.05805)

### 7. **FocalLoss - 损失函数优化**
- **来源**: I-YOLOv11n & MASF-YOLO
- **功能**: 解决小目标检测中的类别不平衡、难易样本问题
- **优势**: 聚焦难样本，提升小目标检测精度
- **参考**: [Focal Loss 原始论文](https://arxiv.org/abs/1708.02002)

### 8. **接口兼容性**
- **来源**: Ultralytics YOLOv11 API
- **功能**: 与 Ultralytics YOLOv11 API 接口兼容，支持训练、推理和 YAML 配置
- **优势**: 无缝集成到现有 YOLO 训练流程
- **参考**: [Ultralytics YOLOv11 官方文档](https://ultralytics.com/blog/yolo11)

## 🏗️ 模型架构

```
输入图像
    ↓
BackboneWithSmallBranch (双分支主干)
    ├── 主分支: 标准特征提取路径
    └── 小目标分支: 高分辨率特征路径 (SmallObjectBranch)
    ↓
CrossScaleAttention (跨尺度注意力融合)
    ├── LocalAttention (局部注意力 + 相对位置编码)
    └── 特征对齐与融合
    ↓
检测头 (Detection Head)
    └── 输出: 检测结果 + 蒸馏损失 (可选)
```

### 核心模块说明

- **GhostModule**: 轻量级卷积模块，通过 Ghost 操作减少参数量
- **SmallObjectBranch**: 基于 GhostModule 的小目标专用分支
- **BackboneWithSmallBranch**: 包含主分支和小目标分支的双路径主干网络
- **LocalAttention**: 简化的 Swin 注意力机制，带相对位置编码
- **CrossScaleAttention**: 跨尺度特征融合模块
- **YOLOv11SmallObjectDetector**: 完整的小目标检测模型，支持知识蒸馏

## 📦 安装与使用

### 环境要求

```bash
torch >= 1.8.0
ultralytics >= 8.0.0
```

### 快速开始

1. **模型初始化**

```python
from ultralytics import YOLO

# 使用 YAML 配置文件
model = YOLO('v11-small.yaml')

# 或直接使用模型名称
model = YOLO('yolov11_smallobject')
```

2. **训练模型**

```python
model.train(
    data='v11-data.yaml',
    epochs=200,
    batch=16,
    imgsz=640,
    optimizer='SGD',
    amp=True,
    project='runs/train',
    name='exp'
)
```

3. **配置文件示例 (v11-small.yaml)**

```yaml
# Parameters
nc: 80   # number of classes

# Backbone
backbone:
  - [custom, 1, YOLOv11SmallObjectDetector]

# Head (minimal head required for parsing)
head:
  - [-1, 1, Detect, [nc]]  # Detect layer
```

## 📚 参考文献

1. **Caiping Xi et al.** "I-YOLOv11n: A Lightweight and Efficient Small Target Detection Framework for UAV Aerial Images." Sensors 25(15):4857, 2025

2. **Chengcheng Wang et al.** "CF-YOLO for small target detection in drone imagery based on YOLOv11 algorithm." Scientific Reports 15:16741, 2025

3. **Liugang Lu et al.** "MASF-YOLO: An Improved YOLOv11 Network for Small Object Detection on Drone View." arXiv:2504.18136, 2025

4. **Yao Xiao et al.** "FBRT-YOLO: Faster and Better for Real-Time Aerial Image Detection." AAAI Conference on Artificial Intelligence, 2025

5. **Ultralytics.** "All you need to know about YOLO11 and its applications." Ultralytics Blog, Oct 4, 2024

## 🔧 技术细节

### 模块对应关系

| 模块名称 | 功能描述 | 对应代码类 |
|---------|---------|-----------|
| Ghost模块 | 高效特征提取，减少参数量 | `GhostModule` |
| 小目标分支 | 提取和增强小目标细节 | `SmallObjectBranch` |
| 主分支 + 小分支 | 同时提取主分支和小目标路径特征 | `BackboneWithSmallBranch` |
| 局部注意力 + 相对位置编码 | 增强小目标空间关系建模 | `LocalAttention` |
| 跨尺度注意力融合 | 融合主分支和小分支特征 | `CrossScaleAttention` |
| 检测头 | 输出小目标检测结果 | `self.head` in `YOLOv11SmallObjectDetector` |
| 特征损失 + 输出损失 | 教师-学生模型对小目标的表征能力 | `loss_feat, loss_output` |

### 设计思想关联

1. **LocalAttention + 位置编码**: 借鉴 Swin Transformer/ViT 的注意力机制和位置编码，增强模型对局部空间关系的建模能力，提升小目标位置的敏感性

2. **CrossScaleAttention 融合模块**: 借鉴 CrossViT/TokenFusion 的多尺度 token 融合机制，通过跨尺度注意力融合，实现细节特征与全局特征的互补

3. **教师蒸馏 loss_feat**: 借鉴 DistillDet/DETR-KD 的中间层特征蒸馏方式，通过特征对齐实现知识迁移，增强学生模型的特征提取能力

4. **输出蒸馏 loss_output**: 借鉴 KD-BERT/TinyViT 的 KL 散度思想，通过预测分布对齐保留细节信息，提升小目标检测精度

5. **FocalLoss**: 借鉴 RetinaNet/GLIP/Grounded-ViT 的 Focal Loss 在类别不平衡场景下的应用，对难样本进行强化，减少背景误识别

6. **GhostModule 轻量化设计**: 借鉴 MobileFormer/TinyViT 的大模型压缩思路，使用 Ghost 操作 + 深度可分离卷积实现模型压缩和特征表示

## 📝 代码位置

- **模型实现**: `/ultralytics/models/yolo/yolov11_smallobject.py`
- **集成代码**: `/ultralytics/nn/tasks.py` (DetectionModel 类中已添加支持)
- **配置文件**: `/v11-small.yaml`
- **训练脚本**: `/train.py`

## 🚀 训练命令

```bash
# 使用 YAML 配置文件训练
python train.py

# 或使用命令行
yolo detect train cfg=v11-small.yaml data=v11-data.yaml model=yolov11_smallobject
```

## 📊 性能特点

- ✅ **轻量化**: 通过 Ghost 模块减少参数量和计算量
- ✅ **高精度**: 多尺度融合和注意力机制提升小目标检测精度
- ✅ **易集成**: 完全兼容 Ultralytics YOLOv11 API
- ✅ **可扩展**: 支持知识蒸馏，可进一步提升性能

## 🤝 贡献

本项目整合了多个前沿研究成果，感谢所有相关论文作者的开源贡献。

## 📄 许可证

本项目基于 Ultralytics YOLOv11，遵循相应的开源许可证。

---

**注意**: 本模型专门针对航拍小目标检测场景优化，在无人机图像、卫星图像等小目标密集场景中表现优异。

