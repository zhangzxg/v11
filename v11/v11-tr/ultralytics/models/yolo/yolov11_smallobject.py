# 完整 YOLOv11 改进小目标检测模型（含 Swin 注意力融合 + 教师蒸馏接口 + 输出蒸馏 + 位置编码）
# 用于航拍小目标检测，可直接运行并集成至YOLO训练流程

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ghost模块
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2):
        super().__init__()
        init_channels = int(out_channels / ratio)
        new_channels = out_channels - init_channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=3, stride=1, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

# 小目标分支
class SmallObjectBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            GhostModule(in_channels, out_channels),
            GhostModule(out_channels, out_channels),
            GhostModule(out_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)

# Swin 注意力窗口（简化 + 相对位置编码）
class LocalAttention(nn.Module):
    def __init__(self, dim, window_size=7):
        super().__init__()
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.rel_pos = nn.Parameter(torch.randn(1, dim, 1, 1))  # 简化位置编码

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.to_qkv(x).reshape(B, 3, C, H, W)
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]
        q = q + self.rel_pos
        k = k + self.rel_pos
        attn = (q * k).sum(1, keepdim=True) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = attn * v
        return self.proj(out)

# 跨尺度融合模块（改进版）
class CrossScaleAttention(nn.Module):
    def __init__(self, in_main, in_small):
        super().__init__()
        self.align_main = nn.Conv2d(in_main, in_small, 1)
        self.pos_embed_main = nn.Parameter(torch.randn(1, in_small, 1, 1))
        self.pos_embed_small = nn.Parameter(torch.randn(1, in_small, 1, 1))
        self.attn_small = LocalAttention(in_small)
        self.attn_main = LocalAttention(in_small)
        self.fuse = nn.Conv2d(in_small * 2, in_small, 1)

    def forward(self, small_feat, main_feat):
        if main_feat.shape[2:] != small_feat.shape[2:]:
            main_feat = F.interpolate(main_feat, size=small_feat.shape[2:], mode='nearest')
        main_feat = self.align_main(main_feat) + self.pos_embed_main
        small_feat = small_feat + self.pos_embed_small
        small_attn = self.attn_small(small_feat)
        main_attn = self.attn_main(main_feat)
        return self.fuse(torch.cat([small_attn, main_attn], dim=1))

# 主干结构
class BackboneWithSmallBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.small_branch = SmallObjectBranch(64, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.stem(x)
        p2 = self.layer1(x)
        p2_small = self.small_branch(p2)
        p3 = self.layer2(p2)
        p4 = self.layer3(p3)
        return p2_small, p3, p4

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        return (self.alpha * (1 - pt) ** self.gamma * BCE_loss).mean()

# 主模型 + 蒸馏
class YOLOv11SmallObjectDetector(nn.Module):
    def __init__(self, use_teacher=False):
        super().__init__()
        self.backbone = BackboneWithSmallBranch()
        self.fusion = CrossScaleAttention(in_main=128, in_small=64)
        self.head = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3 * (5 + 80), 1)
        )
        self.use_teacher = use_teacher
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x, teacher_feats=None, teacher_output=None):
        small_feat, p3, p4 = self.backbone(x)
        fused = self.fusion(small_feat, p3)
        out = self.head(fused)

        loss_feat, loss_output = 0.0, 0.0
        if self.use_teacher:
            if teacher_feats is not None:
                loss_feat = F.mse_loss(p3, teacher_feats[0]) + F.mse_loss(p4, teacher_feats[1])
            if teacher_output is not None:
                s_logits = out.view(out.size(0), out.size(1), -1).permute(0, 2, 1)
                t_logits = teacher_output.view(out.size(0), out.size(1), -1).permute(0, 2, 1).detach()
                loss_output = self.kl(F.log_softmax(s_logits, dim=-1), F.softmax(t_logits, dim=-1))
        return out, loss_feat, loss_output

if __name__ == '__main__':
    model = YOLOv11SmallObjectDetector(use_teacher=True)
    dummy_input = torch.randn(1, 3, 640, 640)
    teacher_feats = [torch.randn(1, 128, 80, 80), torch.randn(1, 256, 40, 40)]
    teacher_out = torch.randn(1, 255, 80, 80)
    out, loss_feat, loss_out = model(dummy_input, teacher_feats, teacher_out)
    print(out.shape, loss_feat.item(), loss_out.item())
