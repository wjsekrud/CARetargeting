import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def compute_parent_joint_offsets(vertices, skinning_weights, joint_positions):
    """
    Compute offsets from each vertex to its parent joint
    Args:
        vertices: [B, N, 3] vertex positions
        skinning_weights: [B, N, J] skinning weights
        joint_positions: [B, J, 3] joint positions
    Returns:
        offsets: [B, N, 3] offset vectors from parent joints to vertices
        parent_indices: [B, N] indices of parent joints
    """
    # Find parent joint indices (joint with maximum weight for each vertex)
    parent_indices = torch.argmax(skinning_weights, dim=2)  # [B, N]
    
    # Gather parent joint positions
    batch_size, num_vertices = vertices.shape[:2]
    batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, num_vertices)
    vertex_indices = torch.arange(num_vertices).view(1, -1).expand(batch_size, -1)
    
    parent_positions = joint_positions[batch_indices, parent_indices]  # [B, N, 3]
    
    # Compute offsets
    offsets = vertices - parent_positions
    
    return offsets, parent_indices

class VertexEncoder(nn.Module):
    def __init__(self, num_joints, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Input dimensions: position (3) + skinning weights (num_joints) + parent offset (3)
        input_dim = 3 + num_joints + 3
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
    
    def forward(self, vertices, skinning_weights, parent_offsets):
        # Concatenate all inputs
        x = torch.cat([vertices, skinning_weights, parent_offsets], dim=-1)
        x = x.view(-1, x.size(-1))
        features = self.mlp(x)
        features = features.view(*vertices.shape[:-1], -1)
        global_features = torch.max(features, dim=1)[0]
        return features, global_features

class VertexDecoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 3)  # Only reconstruct vertex positions
        )
    
    def forward(self, features):
        x = features.view(-1, features.size(-1))
        vertices = self.mlp(x)
        vertices = vertices.view(*features.shape[:-1], 3)
        return vertices

class MeshAutoEncoder(nn.Module):
    def __init__(self, num_joints, feature_dim=128):
        super().__init__()
        self.encoder = VertexEncoder(num_joints, feature_dim)
        self.decoder = VertexDecoder(feature_dim)
    
    def forward(self, vertices, skinning_weights, parent_offsets):
        features, global_features = self.encoder(vertices, skinning_weights, parent_offsets)
        reconstructed_vertices = self.decoder(features)
        return features, reconstructed_vertices

class FeatureLearningFramework:
    def __init__(self, num_joints, feature_dim=128, learning_rate=1e-4):
        self.model = MeshAutoEncoder(num_joints, feature_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def compute_losses(self, vertices, skinning_weights, parent_offsets, features, reconstructed_vertices):
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed_vertices, vertices)
        
        # Feature consistency loss
        # Similar vertices (based on skinning weights and parent joints) should have similar features
        skinning_similarity = torch.bmm(skinning_weights, skinning_weights.transpose(1, 2))
        
        # Add parent joint similarity (vertices with same parent should have more similar features)
        parent_indices = torch.argmax(skinning_weights, dim=2)
        parent_similarity = (parent_indices.unsqueeze(-1) == parent_indices.unsqueeze(1)).float()
        
        # Combine similarities
        vertex_similarity = (skinning_similarity + parent_similarity) / 2
        feature_similarity = torch.bmm(features, features.transpose(1, 2))
        consistency_loss = F.mse_loss(feature_similarity, vertex_similarity)
        
        # Feature norm loss
        feature_norm_loss = torch.mean(torch.norm(features, dim=2))
        
        total_loss = reconstruction_loss #+ 0.1 * consistency_loss + 0.01 * feature_norm_loss
        
        return {
            'total': total_loss,
            'reconstruction': reconstruction_loss,
            'consistency': 0,
            'feature_norm': 0
        }
    
    def train_step(self, vertices, skinning_weights, joint_positions):
        self.optimizer.zero_grad()
        
        # Compute parent joint offsets
        parent_offsets, _ = compute_parent_joint_offsets(vertices, skinning_weights, joint_positions)
        
        # Forward pass
        features, reconstructed_vertices = self.model(vertices, skinning_weights, parent_offsets)
        
        # Compute losses
        losses = self.compute_losses(vertices, skinning_weights, parent_offsets, 
                                   features, reconstructed_vertices)
        
        # Backward pass
        losses['total'].backward()
        self.optimizer.step()
        
        return losses
    
    def extract_features(self, vertices, skinning_weights, joint_positions):
        """Use this after training to extract features using only the encoder"""
        with torch.no_grad():
            parent_offsets, _ = compute_parent_joint_offsets(vertices, skinning_weights, joint_positions)
            features, global_features = self.model.encoder(vertices, skinning_weights, parent_offsets)
        return features, global_features

def test_framework():
    # Test parameters
    batch_size = 2
    num_vertices = 1024
    num_joints = 24
    
    # Create sample data
    vertices = torch.randn(batch_size, num_vertices, 3)
    skinning_weights = torch.randn(batch_size, num_vertices, num_joints)
    skinning_weights = F.softmax(skinning_weights, dim=-1)  # Normalize weights
    joint_positions = torch.randn(batch_size, num_joints, 3)
    
    # Initialize framework
    framework = FeatureLearningFramework(num_joints)
    
    # Test forward pass
    losses = framework.train_step(vertices, skinning_weights, joint_positions)
    
    print("Test forward pass completed")
    print("Losses:", {k: v.item() for k, v in losses.items()})
    
    # Test feature extraction
    features, global_features = framework.extract_features(vertices, skinning_weights, joint_positions)
    print(f"\nExtracted features shape: {features.shape}")
    print(f"Global features shape: {global_features.shape}")

if __name__ == "__main__":
    test_framework()