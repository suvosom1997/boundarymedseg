# ==================== losses.py ====================

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        if pred.dim() == 4 and pred.size(1) > 1:  # Multi-class
            pred = F.softmax(pred, dim=1)
            if target.dim() == 3:
                target = F.one_hot(target.long(), num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        else:
            pred = torch.sigmoid(pred)
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for segmentation"""
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        if pred.dim() == 4 and pred.size(1) > 1:
            pred = F.softmax(pred, dim=1)
            if target.dim() == 3:
                target = F.one_hot(target.long(), num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        else:
            pred = torch.sigmoid(pred)

        TP = (pred * target).sum(dim=(2, 3))
        FP = ((1 - target) * pred).sum(dim=(2, 3))
        FN = (target * (1 - pred)).sum(dim=(2, 3))

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return ((1 - tversky) ** self.gamma).mean()

class BoundaryLoss(nn.Module):
    """Boundary detection loss"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def generate_boundaries(self, seg_mask):
        if seg_mask.dim() == 4 and seg_mask.size(1) > 1:
            seg_mask = seg_mask.argmax(dim=1)
        elif seg_mask.dim() == 4:
            seg_mask = seg_mask.squeeze(1)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=seg_mask.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=seg_mask.device).view(1, 1, 3, 3)

        edges_x = F.conv2d(seg_mask.float().unsqueeze(1), sobel_x, padding=1)
        edges_y = F.conv2d(seg_mask.float().unsqueeze(1), sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)

        return torch.sigmoid(edges).squeeze(1)

    def forward(self, pred_boundaries, target_seg):
        target_boundaries = self.generate_boundaries(target_seg)
        pred_boundaries = pred_boundaries.squeeze(1)
        return self.bce(pred_boundaries, target_boundaries)

class CombinedLoss(nn.Module):
    """Combined loss for BoundaryMedSeg using Dice + Focal Tversky + Boundary"""
    def __init__(self, seg_weight=1.0, boundary_weight=0.2):
        super().__init__()
        self.seg_weight = seg_weight
        self.boundary_weight = boundary_weight
        self.dice_loss = DiceLoss()
        self.focal_tversky_loss = FocalTverskyLoss()
        self.boundary_loss_fn = BoundaryLoss()

    def forward(self, pred_seg, pred_boundaries, target_seg):
        dice = self.dice_loss(pred_seg, target_seg)
        focal_tversky = self.focal_tversky_loss(pred_seg, target_seg)
        boundary = self.boundary_loss_fn(pred_boundaries, target_seg)

        seg_loss = dice + focal_tversky
        total_loss = self.seg_weight * seg_loss + self.boundary_weight * boundary

        return total_loss, dice, focal_tversky, boundary
