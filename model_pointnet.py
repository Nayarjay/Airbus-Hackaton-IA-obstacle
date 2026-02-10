import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetSeg(nn.Module):
    def __init__(self, k=5, feature_transform=True):
        super(PointNetSeg, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        
        # Shared MLP (PointNet Encoder)
        # Input (3, N) -> (64, N)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        
        # (64, N) -> (128, N)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        
        # (128, N) -> (1024, N) (Global Feature Generation)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # T-Nets (Optional but recommended)
        self.stn = STN3d()
        if self.feature_transform:
            self.fstn = STNkd(k=64)

        # Segmentation Decoder
        # Concatenate Local (64) + Global (1024) Features -> 1088
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.bn6 = nn.BatchNorm1d(128)
        
        # Output: k classes
        self.conv7 = torch.nn.Conv1d(128, k, 1)

    def forward(self, x):
        n_pts = x.size()[2]
        
        # 1. Input Transform
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        
        # 2. First MLP (Local Features)
        x = F.relu(self.bn1(self.conv1(x))) # (64, N)
        
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
            
        pointfeat = x # Store local features (64, N)
        
        # 3. Global Features
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) # (1024, N)
        x = torch.max(x, 2, keepdim=True)[0] # (1024, 1) Global Max Pooling
        x = x.view(-1, 1024)
        
        # 4. Concatenate Global + Local
        # Expand global feature to (1024, N)
        global_feat_expanded = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        concat = torch.cat([pointfeat, global_feat_expanded], 1) # (1088, N)
        
        # 5. Segmentation Head
        x = F.relu(self.bn4(self.conv4(concat)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.conv7(x) # (k, N)
        
        x = x.transpose(2, 1).contiguous() # (Batch, N, k)
        x = F.log_softmax(x, dim=-1)
        
        return x, trans_feat

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3, dtype=x.dtype, device=x.device).view(1, 9).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()
        self.k = k

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k*self.k).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
