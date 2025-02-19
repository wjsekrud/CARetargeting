import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    """Transform Net for input transformation"""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        
        # Shared MLP
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # MLP for transform matrix
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        # Initialize last layer with zeros + identity matrix
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).view(-1))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # (batch_size, k, num_points)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Reshape to transform matrix
        identity = torch.eye(self.k).view(1, self.k*self.k).repeat(batch_size, 1)
        if x.is_cuda:
            identity = identity.cuda()
        x = x + identity
        x = x.view(-1, self.k, self.k)
        
        return x

class VertexFeatureExtractor(nn.Module):
    """Feature extractor for mesh vertices based on PointNet architecture"""
    def __init__(self, num_joints):
        super().__init__()
        self.num_joints = num_joints
        
        # Input transform net (3D coordinates)
        self.stn = TNet(k=3)
        
        # Feature transform net
        self.fstn = TNet(k=64)
        
        # Skinning weights processing
        self.skinning_mlp = nn.Sequential(
            nn.Linear(num_joints, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Shared MLP layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Final MLP for per-vertex features
        self.mlp = nn.Sequential(
            nn.Linear(1024 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward(self, vertices, skinning_weights):
        batch_size = vertices.size(0)
        num_points = vertices.size(1)
        
        # Transform input vertices
        trans = self.stn(vertices.transpose(2, 1))
        vertices = torch.bmm(vertices, trans)
        
        # Process vertices through shared MLP
        x = vertices.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Feature transform
        trans_feat = self.fstn(x)
        x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)
        
        # Continue with shared MLP
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Process skinning weights
        skinning_features = self.skinning_mlp(skinning_weights)
        
        # Global feature vector
        global_feat = torch.max(x, 2, keepdim=True)[0]
        global_feat = global_feat.view(-1, 1024)
        
        # Concatenate global features with skinning features
        combined_features = torch.cat([
            global_feat.unsqueeze(1).repeat(1, num_points, 1),
            skinning_features
        ], dim=2)
        
        # Final per-vertex features
        vertex_features = self.mlp(combined_features.view(-1, 1024 + 64))
        vertex_features = vertex_features.view(batch_size, num_points, -1)
        
        return vertex_features

# Example usage
def test_model():
    # Create sample data
    batch_size = 2
    num_points = 1024
    num_joints = 24
    
    vertices = torch.randn(batch_size, num_points, 3)
    skinning_weights = torch.randn(batch_size, num_points, num_joints)
    skinning_weights = F.softmax(skinning_weights, dim=-1)  # Normalize weights
    
    # Initialize model
    model = VertexFeatureExtractor(num_joints=num_joints)
    
    # Forward pass
    features = model(vertices, skinning_weights)
    print(f"Input vertices shape: {vertices.shape}")
    print(f"Input skinning weights shape: {skinning_weights.shape}")
    print(f"Output features shape: {features.shape}")
    
    return features

if __name__ == "__main__":
    test_model()