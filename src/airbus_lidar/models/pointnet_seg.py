from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetBackbone(nn.Module):
    """
    Backbone minimal PointNet:
    - per-point MLP via Conv1d(1x1)
    - global feature via max pooling
    """
    def __init__(self, in_channels: int = 4, feat_dim: int = 1024):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, feat_dim, 1)
        self.bn3 = nn.BatchNorm1d(feat_dim)

        self.feat_dim = feat_dim
        self.per_point_dim = 128

    def forward(self, x: torch.Tensor):
        """
        x: (B,C,N)
        returns:
          per_point: (B,128,N)
          global_feat: (B,feat_dim,1)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        per_point = F.relu(self.bn2(self.conv2(x)))  # (B,128,N)
        glob = F.relu(self.bn3(self.conv3(per_point)))  # (B,feat_dim,N)
        global_feat = torch.max(glob, dim=2, keepdim=True)[0]  # (B,feat_dim,1)
        return per_point, global_feat

    @torch.no_grad()
    def forward_per_point_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,N)
        returns per_point: (B,128,N)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        per_point = F.relu(self.bn2(self.conv2(x)))
        return per_point


class PointNetSeg(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, feat_dim: int = 1024):
        super().__init__()
        self.backbone = PointNetBackbone(in_channels=in_channels, feat_dim=feat_dim)

        # head segmentation
        self.conv1 = nn.Conv1d(self.backbone.per_point_dim + feat_dim, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv_out = nn.Conv1d(128, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard train forward.
        x: (B,C,N)
        returns logits: (B,num_classes,N)
        """
        per_point, global_feat = self.backbone(x)
        global_rep = global_feat.expand(-1, -1, per_point.size(-1))
        feat = torch.cat([per_point, global_rep], dim=1)

        feat = self.drop1(F.relu(self.bn1(self.conv1(feat))))
        feat = self.drop2(F.relu(self.bn2(self.conv2(feat))))
        feat = F.relu(self.bn3(self.conv3(feat)))
        logits = self.conv_out(feat)
        return logits

    @torch.no_grad()
    def predict_full_cloud(
        self,
        x_full: torch.Tensor,
        x_global: torch.Tensor,
        chunk_size: int = 65536,
    ) -> torch.Tensor:
        """
        Full labeling avec global feature calculée sur un sous-échantillon.
        - x_full: (1,C,Nfull)
        - x_global: (1,C,Ng)
        return logits_full: (1,num_classes,Nfull)
        """
        self.eval()
        _, global_feat = self.backbone(x_global)  # (1,feat_dim,1)

        N = x_full.size(-1)
        logits_list = []

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = x_full[:, :, start:end]  # (1,C,nc)

            per_point = self.backbone.forward_per_point_only(x_chunk)  # (1,128,nc)
            global_rep = global_feat.expand(-1, -1, per_point.size(-1))
            feat = torch.cat([per_point, global_rep], dim=1)

            feat = F.relu(self.bn1(self.conv1(feat)))
            feat = F.relu(self.bn2(self.conv2(feat)))
            feat = F.relu(self.bn3(self.conv3(feat)))
            logits = self.conv_out(feat)  # (1,num_classes,nc)
            logits_list.append(logits)

        return torch.cat(logits_list, dim=-1)
