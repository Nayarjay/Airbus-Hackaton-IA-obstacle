# pointnet_seg.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    """Transformation Network (T-Net) pour aligner l'entrée ou les features."""
    def __init__(self, k=3):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x: (B, k, N)
        B = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))          # (B,1024,N)

        x = torch.max(x, 2)[0]                       # (B,1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)                              # (B, k*k)

        # init proche identité
        eye = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(B, 1)
        x = x + eye
        x = x.view(-1, self.k, self.k)               # (B,k,k)
        return x

class PointNetEncoder(nn.Module):
    """
    Encoder PointNet:
    - input transform (3x3)
    - feature transform (64x64) (optionnel mais utile)
    - global feature (1024)
    """
    def __init__(self, feature_transform=True):
        super().__init__()
        self.feature_transform = feature_transform

        self.input_tnet = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.feature_transform:
            self.feature_tnet = TNet(k=64)

    def forward(self, x):
        # x: (B, N, 3)
        x = x.transpose(2, 1)                        # (B,3,N)

        # input transform
        trans = self.input_tnet(x)                   # (B,3,3)
        x = x.transpose(2, 1)                        # (B,N,3)
        x = torch.bmm(x, trans)                      # (B,N,3)
        x = x.transpose(2, 1)                        # (B,3,N)

        # features
        x = F.relu(self.bn1(self.conv1(x)))          # (B,64,N)

        # feature transform
        trans_feat = None
        if self.feature_transform:
            trans_feat = self.feature_tnet(x)        # (B,64,64)
            x = x.transpose(2, 1)                    # (B,N,64)
            x = torch.bmm(x, trans_feat)             # (B,N,64)
            x = x.transpose(2, 1)                    # (B,64,N)

        pointfeat = x                                # (B,64,N)

        x = F.relu(self.bn2(self.conv2(x)))          # (B,128,N)
        x = self.bn3(self.conv3(x))                  # (B,1024,N)

        global_feat = torch.max(x, 2, keepdim=True)[0]  # (B,1024,1)
        global_feat = global_feat.repeat(1, 1, pointfeat.size(2))  # (B,1024,N)

        return global_feat, pointfeat, trans_feat

class PointNetSeg(nn.Module):
    """
    Segmentation head:
    concat(global_feat 1024, pointfeat 64) -> 1088
    output logits (B,N,C)
    """
    def __init__(self, num_classes=4, feature_transform=True):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = PointNetEncoder(feature_transform=feature_transform)

        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=0.3)

    def forward(self, x):
        # x: (B,N,3)
        global_feat, pointfeat, _trans_feat = self.encoder(x)      # (B,1024,N) & (B,64,N)
        x = torch.cat([global_feat, pointfeat], dim=1)             # (B,1088,N)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)                                          # (B,C,N)

        x = x.transpose(2, 1)                                      # (B,N,C)
        return x
