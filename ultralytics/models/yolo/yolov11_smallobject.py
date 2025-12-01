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
    def __init__(self, in_channels, out_channels, use_ghost=True):
        super().__init__()
        if use_ghost:
            self.block = nn.Sequential(
                GhostModule(in_channels, out_channels),
                GhostModule(out_channels, out_channels),
                GhostModule(out_channels, out_channels)
            )
        else:
            # 使用标准卷积替代Ghost模块
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)

# Swin 注意力窗口（简化 + 相对位置编码）
class LocalAttention(nn.Module):
    def __init__(self, dim, window_size=7, use_attention=True, use_pos_encoding=True):
        super().__init__()
        self.use_attention = use_attention
        self.use_pos_encoding = use_pos_encoding
        if use_attention:
            self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
            self.proj = nn.Conv2d(dim, dim, 1)
            if use_pos_encoding:
                self.rel_pos = nn.Parameter(torch.randn(1, dim, 1, 1))  # 简化位置编码
            else:
                self.rel_pos = None
        else:
            # 使用标准卷积替代注意力机制
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.use_attention:
            B, C, H, W = x.shape
            qkv = self.to_qkv(x).reshape(B, 3, C, H, W)
            q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]
            if self.use_pos_encoding and self.rel_pos is not None:
                q = q + self.rel_pos
                k = k + self.rel_pos
            attn = (q * k).sum(1, keepdim=True) / (C ** 0.5)
            attn = F.softmax(attn, dim=-1)
            out = attn * v
            return self.proj(out)
        else:
            return self.conv(x)

# 跨尺度融合模块（改进版）
class CrossScaleAttention(nn.Module):
    def __init__(self, in_main, in_small, use_attention=True, use_pos_encoding=True):
        super().__init__()
        self.use_attention = use_attention
        self.use_pos_encoding = use_pos_encoding
        self.align_main = nn.Conv2d(in_main, in_small, 1)
        if use_pos_encoding:
            self.pos_embed_main = nn.Parameter(torch.randn(1, in_small, 1, 1))
            self.pos_embed_small = nn.Parameter(torch.randn(1, in_small, 1, 1))
        else:
            self.pos_embed_main = None
            self.pos_embed_small = None
        
        if use_attention:
            self.attn_small = LocalAttention(in_small, use_attention=True, use_pos_encoding=use_pos_encoding)
            self.attn_main = LocalAttention(in_small, use_attention=True, use_pos_encoding=use_pos_encoding)
            self.fuse = nn.Conv2d(in_small * 2, in_small, 1)
        else:
            # 使用简单concat + conv替代注意力融合
            self.fuse = nn.Sequential(
                nn.Conv2d(in_small * 2, in_small, 1, bias=False),
                nn.BatchNorm2d(in_small),
                nn.ReLU(inplace=True)
            )

    def forward(self, small_feat, main_feat):
        if main_feat.shape[2:] != small_feat.shape[2:]:
            main_feat = F.interpolate(main_feat, size=small_feat.shape[2:], mode='nearest')
        main_feat = self.align_main(main_feat)
        if self.use_pos_encoding and self.pos_embed_main is not None:
            main_feat = main_feat + self.pos_embed_main
            small_feat = small_feat + self.pos_embed_small
        
        if self.use_attention:
            small_attn = self.attn_small(small_feat)
            main_attn = self.attn_main(main_feat)
            return self.fuse(torch.cat([small_attn, main_attn], dim=1))
        else:
            return self.fuse(torch.cat([small_feat, main_feat], dim=1))

# 主干结构
class BackboneWithSmallBranch(nn.Module):
    def __init__(self, use_small_branch=True, use_ghost=True):
        super().__init__()
        self.use_small_branch = use_small_branch
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
        if use_small_branch:
            self.small_branch = SmallObjectBranch(64, 64, use_ghost=use_ghost)
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
        if self.use_small_branch:
            p2_small = self.small_branch(p2)
        else:
            # 如果不使用小目标分支，直接使用p2
            p2_small = p2
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
    def __init__(self, use_teacher=False, 
                 use_small_branch=True,      # 消融: 是否使用小目标分支
                 use_ghost=True,             # 消融: 是否使用Ghost模块
                 use_attention=True,         # 消融: 是否使用注意力机制
                 use_pos_encoding=True,      # 消融: 是否使用位置编码
                 use_cross_scale_fusion=True, # 消融: 是否使用跨尺度融合
                 nc=80):                     # 类别数
        super().__init__()
        self.use_teacher = use_teacher
        self.use_small_branch = use_small_branch
        self.use_cross_scale_fusion = use_cross_scale_fusion
        
        # 构建backbone
        self.backbone = BackboneWithSmallBranch(
            use_small_branch=use_small_branch,
            use_ghost=use_ghost
        )
        
        # 构建融合模块
        if use_cross_scale_fusion and use_small_branch:
            self.fusion = CrossScaleAttention(
                in_main=128, 
                in_small=64,
                use_attention=use_attention,
                use_pos_encoding=use_pos_encoding
            )
            fusion_in_channels = 64
        else:
            # 如果不使用跨尺度融合或小目标分支，直接使用主分支特征
            self.fusion = None
            fusion_in_channels = 128
        
        # 构建多尺度检测头
        # 标准 YOLO 使用 3 个尺度: P3/8, P4/16, P5/32
        # 我们使用: P2/4 (small_feat), P3/8 (p3), P4/16 (p4)
        reg_max = 16  # DFL channels
        no = nc + reg_max * 4  # number of outputs per anchor
        
        # 检测头1: 用于 P2/4 尺度 (small_feat, 高分辨率，用于小目标)
        if use_cross_scale_fusion and use_small_branch:
            head1_in_channels = 64  # fused feature from small branch
        else:
            if use_small_branch:
                head1_in_channels = 64  # small_feat
            else:
                # 如果没有小目标分支，需要从 p3 适配
                head1_in_channels = 128  # p3 channels, will be adapted to 64
                self.p2_adapter = nn.Conv2d(128, 64, 1)  # Adapter for p3 to p2 scale
        
        # 检测头2: 用于 P3/8 尺度 (p3, 中等分辨率)
        head2_in_channels = 128  # p3
        
        # 检测头3: 用于 P4/16 尺度 (p4, 低分辨率)
        head3_in_channels = 256  # p4
        
        # 优化检测头以减少内存占用
        # 使用更小的中间通道数，特别是对于高分辨率的 head1
        self.head1 = nn.Sequential(
            nn.Conv2d(head1_in_channels, 64, 3, padding=1),  # 减小到64通道
            nn.ReLU(),
            nn.Conv2d(64, no, 1)  # (nc + reg_max * 4)
        )
        
        self.head2 = nn.Sequential(
            nn.Conv2d(head2_in_channels, 128, 3, padding=1),  # 减小到128通道
            nn.ReLU(),
            nn.Conv2d(128, no, 1)  # (nc + reg_max * 4)
        )
        
        self.head3 = nn.Sequential(
            nn.Conv2d(head3_in_channels, 256, 3, padding=1),  # 减小到256通道
            nn.ReLU(),
            nn.Conv2d(256, no, 1)  # (nc + reg_max * 4)
        )
        
        if use_teacher:
            self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x, teacher_feats=None, teacher_output=None):
        small_feat, p3, p4 = self.backbone(x)
        
        # 生成多尺度检测输出
        # 尺度1: P2/4 (small_feat 或融合特征) - 高分辨率，用于小目标检测
        if self.use_cross_scale_fusion and self.use_small_branch and self.fusion is not None:
            fused_p2 = self.fusion(small_feat, p3)
            out1 = self.head1(fused_p2)  # P2/4 scale
        else:
            # 如果不使用融合，使用 small_feat 或 p3 的适配
            if self.use_small_branch:
                # 使用 small_feat (已经是 64 通道)
                out1 = self.head1(small_feat)
            else:
                # 如果没有小目标分支，使用 p3 并适配到 P2 尺度
                # 上采样 p3 到更高分辨率 (P2/4 尺度)
                # p3 是 P3/8，需要上采样到 P2/4 (2倍上采样)
                p3_up = F.interpolate(p3, scale_factor=2, mode='nearest')
                # 适配通道数从 128 到 64
                p3_adapted = self.p2_adapter(p3_up)
                out1 = self.head1(p3_adapted)
        
        # 尺度2: P3/8 (p3) - 中等分辨率
        out2 = self.head2(p3)  # P3/8 scale
        
        # 尺度3: P4/16 (p4) - 低分辨率
        out3 = self.head3(p4)  # P4/16 scale
        
        # 返回多尺度特征图列表 (对应标准 YOLO 的 P3, P4, P5)
        outputs = [out1, out2, out3]

        loss_feat, loss_output = 0.0, 0.0
        if self.use_teacher:
            if teacher_feats is not None:
                loss_feat = F.mse_loss(p3, teacher_feats[0]) + F.mse_loss(p4, teacher_feats[1])
            if teacher_output is not None:
                # 对于多尺度输出，使用第一个尺度进行蒸馏
                s_logits = outputs[0].view(outputs[0].size(0), outputs[0].size(1), -1).permute(0, 2, 1)
                t_logits = teacher_output.view(teacher_output.size(0), teacher_output.size(1), -1).permute(0, 2, 1).detach()
                loss_output = self.kl(F.log_softmax(s_logits, dim=-1), F.softmax(t_logits, dim=-1))
        
        # 返回多尺度输出列表，与标准 YOLO 格式一致
        if self.use_teacher:
            return outputs, loss_feat, loss_output
        else:
            return outputs

if __name__ == '__main__':
    model = YOLOv11SmallObjectDetector(use_teacher=True)
    dummy_input = torch.randn(1, 3, 640, 640)
    teacher_feats = [torch.randn(1, 128, 80, 80), torch.randn(1, 256, 40, 40)]
    teacher_out = torch.randn(1, 255, 80, 80)
    out, loss_feat, loss_out = model(dummy_input, teacher_feats, teacher_out)
    print(out.shape, loss_feat.item(), loss_out.item())
