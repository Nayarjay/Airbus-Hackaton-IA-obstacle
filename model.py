import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetSeg(nn.Module):
    def __init__(self, num_classes=5):
        # num_classes = 5 (0: Background, 1: Antenna, 2: Cable, 3: Pole, 4: Turbine)
        super(PointNetSeg, self).__init__()

        # Encodeur (MLP par point)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Décodeur (Segmentation)
        # 1088 = 1024 (Global Feature) + 64 (Local Feature from conv1)
        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, num_classes, 1)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

    def forward(self, x):
        # x shape: (Batch, 3, NumPoints)
        batch_size = x.size(0)
        num_points = x.size(2)

        # --- ENCODER ---
        x1 = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        x2 = F.relu(self.bn2(self.conv2(x1)))  # (B, 128, N)
        x3 = self.bn3(self.conv3(x2))  # (B, 1024, N)

        # Global Max Pooling -> Feature globale de la forme (B, 1024, 1)
        x_global = torch.max(x3, 2, keepdim=True)[0]

        # On répète la feature globale pour chaque point
        x_global_repeated = x_global.repeat(1, 1, num_points)  # (B, 1024, N)

        # Concatenation : Information Globale + Information Locale (x1)
        concat = torch.cat([x_global_repeated, x1], 1)  # (B, 1088, N)

        # --- DECODER ---
        net = F.relu(self.bn4(self.conv4(concat)))
        net = F.relu(self.bn5(self.conv5(net)))
        net = F.relu(self.bn6(self.conv6(net)))

        # Output final : Score pour chaque classe pour chaque point
        output = self.conv7(net)  # (B, NumClasses, N)

        return output