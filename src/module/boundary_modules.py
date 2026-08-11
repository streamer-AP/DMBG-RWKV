import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def generate_boundary_gt(label, num_classes, kernel_size=3):
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    padding = kernel_size // 2
    boundaries = []
    for c in range(1, num_classes):
        mask_c = (label == c).float().unsqueeze(1)
        dilated = F.max_pool2d(mask_c, kernel_size, stride=1, padding=padding)
        eroded = 1.0 - F.max_pool2d(1.0 - mask_c, kernel_size, stride=1, padding=padding)
        boundaries.append((dilated - eroded).clamp(0.0, 1.0))
    return torch.cat(boundaries, dim=1)


class BFH(nn.Module):
    """Boundary-guided Feature Harmonization used by the final decoder."""

    def __init__(self, in_channels, num_fg_classes):
        super().__init__()
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_fg_classes, 1),
        )
        self.gate_proj = nn.Conv2d(num_fg_classes, in_channels, 1, bias=False)
        nn.init.constant_(self.gate_proj.weight, 1.0 / num_fg_classes)
        self.refine_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.refine_conv.weight)

    def forward(self, feat):
        boundary_logit = self.boundary_conv(feat)
        b_gate = torch.sigmoid(boundary_logit)
        b_gate = self.gate_proj(b_gate)
        feat_refined = feat + self.refine_conv(feat * b_gate)
        return feat_refined, boundary_logit
