import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Transformation Network for learning spatial/feature transformations."""

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
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2)[0]  # Global max pooling

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialize as identity matrix
        identity = torch.eye(self.k, device=x.device, dtype=x.dtype)
        identity = identity.view(1, self.k * self.k).repeat(batch_size, 1)

        x = x + identity
        x = x.view(-1, self.k, self.k)

        return x


class PointNetEncoder(nn.Module):
    """PointNet encoder with T-Net transformations."""

    def __init__(self, input_channels=4, feature_transform=True):
        super().__init__()
        self.feature_transform = feature_transform

        self.tnet3 = TNet(k=3)

        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        if feature_transform:
            self.tnet64 = TNet(k=64)

        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        # x: [B, N, input_channels]
        batch_size, n_points, _ = x.size()

        # Extract xyz for spatial transformation
        xyz = x[:, :, :3]  # [B, N, 3]

        # Spatial transformation
        xyz_t = xyz.transpose(2, 1)  # [B, 3, N]
        trans3 = self.tnet3(xyz_t)  # [B, 3, 3]
        xyz = torch.bmm(xyz, trans3)  # [B, N, 3]

        # Reconstruct input with transformed xyz
        if x.size(2) > 3:
            x = torch.cat([xyz, x[:, :, 3:]], dim=2)
        else:
            x = xyz

        x = x.transpose(2, 1)  # [B, C, N]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transformation
        trans64 = None
        if self.feature_transform:
            trans64 = self.tnet64(x)  # [B, 64, 64]
            x = x.transpose(2, 1)  # [B, N, 64]
            x = torch.bmm(x, trans64)
            x = x.transpose(2, 1)  # [B, 64, N]

        point_features = x  # Save for segmentation [B, 64, N]

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Global feature
        global_feature = torch.max(x, 2)[0]  # [B, 1024]

        return global_feature, point_features, trans3, trans64


class PointNetSegmentation(nn.Module):
    """PointNet for semantic segmentation with enhanced decoder."""

    def __init__(self, num_classes=5, input_channels=4, feature_transform=True, enhanced_decoder=True):
        super().__init__()
        self.num_classes = num_classes
        self.feature_transform = feature_transform
        self.enhanced_decoder = enhanced_decoder

        self.encoder = PointNetEncoder(
            input_channels=input_channels,
            feature_transform=feature_transform
        )

        # Segmentation head (enhanced decoder with extra layer)
        if enhanced_decoder:
            # v3: decoder renforcé
            self.conv0 = nn.Conv1d(1088, 1024, 1)  # Extra layer
            self.bn0 = nn.BatchNorm1d(1024)
            self.conv1 = nn.Conv1d(1024, 512, 1)
        else:
            # Original decoder
            self.conv1 = nn.Conv1d(1088, 512, 1)  # 1024 + 64 = 1088

        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # x: [B, N, input_channels]
        batch_size, n_points, _ = x.size()

        global_feature, point_features, trans3, trans64 = self.encoder(x)

        # Expand global feature and concatenate with point features
        global_feature = global_feature.unsqueeze(2).repeat(1, 1, n_points)  # [B, 1024, N]
        concat_features = torch.cat([point_features, global_feature], dim=1)  # [B, 1088, N]

        # Segmentation MLP
        if self.enhanced_decoder:
            x = F.relu(self.bn0(self.conv0(concat_features)))
            x = F.relu(self.bn1(self.conv1(x)))
        else:
            x = F.relu(self.bn1(self.conv1(concat_features)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.conv4(x)  # [B, num_classes, N]

        x = x.transpose(2, 1)  # [B, N, num_classes]

        return x, trans3, trans64


def feature_transform_regularizer(trans):
    """Regularization loss for feature transformation matrix."""
    if trans is None:
        return 0

    d = trans.size(1)
    batch_size = trans.size(0)

    identity = torch.eye(d, device=trans.device, dtype=trans.dtype)
    identity = identity.unsqueeze(0).repeat(batch_size, 1, 1)

    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - identity, dim=(1, 2)))
    return loss


def get_model_size(model):
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = PointNetSegmentation(num_classes=5, input_channels=4)
    print(f"Model parameters: {get_model_size(model):,}")

    # Test forward pass
    batch_size = 2
    n_points = 1024
    x = torch.randn(batch_size, n_points, 4)

    output, trans3, trans64 = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trans3 shape: {trans3.shape}")
    print(f"Trans64 shape: {trans64.shape}")

    # Test regularization loss
    reg_loss = feature_transform_regularizer(trans64)
    print(f"Feature transform regularization loss: {reg_loss:.4f}")
