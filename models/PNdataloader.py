import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class MeshVertex:
    """Simplified version of NewVertex for our specific needs"""
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.bone_weights = []
        
    @staticmethod
    def load_from_txt(file_path: str) -> Tuple[list, list, float, float]:
        """Load vertex data from text file"""
        vertices = []
        triangles = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            height = float(lines[0].split(":")[1])
            max_geodesic = float(lines[1].split(":")[1])
            vertex_count = int(lines[2].split(":")[1])
            
            # Parse vertices
            vertex_lines = lines[5:5 + vertex_count]
            for line in vertex_lines:
                if not line.strip():
                    continue
                position, bone_weight, _ = line.strip().split('|')
                
                vertex = MeshVertex()
                vertex.position = [float(x) * 0.01 for x in position.split(',')]
                vertex.bone_weights = [
                    {
                        'bone_index': int(bw.split(':')[0]),
                        'weight': float(bw.split(':')[1])
                    } for bw in bone_weight.split(',') if bw
                ]
                vertices.append(vertex)
            
            # Parse triangles
            triangle_lines = lines[6 + vertex_count:]
            for line in triangle_lines:
                if not line.strip():
                    continue
                triangles.append(list(map(int, line.strip().split(','))))
        
        return vertices, triangles, height, max_geodesic

class FixedSizeMeshDataset(Dataset):
    def __init__(self, mesh_files: list, max_vertices: int, num_joints: int = 22):
        """
        Args:
            mesh_files: List of paths to mesh files
            max_vertices: Maximum number of vertices to support
            num_joints: Number of joints in the skeleton
        """
        self.mesh_files = mesh_files
        self.max_vertices = max_vertices
        self.num_joints = num_joints
        self.meshes = []
        
        # Find the actual maximum number of vertices
        self.actual_max_vertices = 0
        
        # Load all mesh files
        for file_path in mesh_files:
            print(file_path)
            vertices, triangles, height, max_geodesic = MeshVertex.load_from_txt(file_path)
            self.actual_max_vertices = max(self.actual_max_vertices, len(vertices))
            
            # Convert to padded numpy arrays
            vertex_positions = np.zeros((max_vertices, 3))
            skinning_weights = np.zeros((max_vertices, num_joints))
            vertex_mask = np.zeros(max_vertices, dtype=bool)
            
            # Fill arrays with actual data
            num_vertices = min(len(vertices), max_vertices)
            for i in range(num_vertices):
                vertex_positions[i] = vertices[i].position
                for weight in vertices[i].bone_weights:
                    skinning_weights[i][weight['bone_index']] = weight['weight']
                vertex_mask[i] = True
            
            # Pad triangles with -1 for invalid indices
            padded_triangles = np.full((max_vertices, 3), -1)
            num_triangles = min(len(triangles), max_vertices)
            padded_triangles[:num_triangles] = triangles[:num_triangles]
            
            self.meshes.append({
                'positions': vertex_positions,
                'weights': skinning_weights,
                'triangles': padded_triangles,
                'mask': vertex_mask,
                'num_vertices': num_vertices,
                'height': height,
                'max_geodesic': max_geodesic
            })
            
        if self.actual_max_vertices > max_vertices:
            print(f"Warning: Some meshes have more vertices ({self.actual_max_vertices}) "
                  f"than the specified maximum ({max_vertices}). These will be truncated.")
    
    def __len__(self) -> int:
        return len(self.meshes)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        mesh = self.meshes[idx]
        
        # Convert to torch tensors
        positions = torch.FloatTensor(mesh['positions'])
        weights = torch.FloatTensor(mesh['weights'])
        mask = torch.BoolTensor(mesh['mask'])
        triangles = torch.LongTensor(mesh['triangles'])
        
        return positions, weights, mask, triangles, mesh['num_vertices']
    
    def get_joint_positions(self, idx: int) -> Optional[np.ndarray]:
        """Returns joint positions based on weighted vertex positions"""
        mesh = self.meshes[idx]
        positions = mesh['positions']
        weights = mesh['weights']
        mask = mesh['mask']
        
        # Only use valid vertices for joint position calculation
        valid_positions = positions[mask]
        valid_weights = weights[mask]
        
        joint_positions = np.zeros((self.num_joints, 3))
        for j in range(self.num_joints):
            joint_weights = valid_weights[:, j]
            if np.sum(joint_weights) > 0:
                joint_positions[j] = np.average(valid_positions, weights=joint_weights, axis=0)
            else:
                joint_positions[j] = np.mean(valid_positions, axis=0)
        
        return joint_positions

def create_mesh_dataloader(
    mesh_dir: str,
    max_vertices: int,
    batch_size: int = 1,
    num_joints: int = 22,
    file_pattern: str = "*_clean_vertices.txt"
) -> Tuple[DataLoader, FixedSizeMeshDataset]:
    """
    Create a dataloader for fixed-size mesh data
    
    Args:
        mesh_dir: Directory containing mesh files
        max_vertices: Maximum number of vertices to support
        batch_size: Batch size for the dataloader
        num_joints: Number of joints in the skeleton
        file_pattern: Pattern to match mesh files
    """
    mesh_files = list(Path(mesh_dir).glob(file_pattern))
    if not mesh_files:
        raise ValueError(f"No mesh files found in {mesh_dir} matching pattern {file_pattern}")
    
    dataset = FixedSizeMeshDataset(mesh_files, max_vertices, num_joints)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    return dataloader, dataset

# Usage example
if __name__ == "__main__":
    # Example usage with fixed size tensors
    dataloader, dataset = create_mesh_dataloader(
        mesh_dir="./dataset/vertexes",
        max_vertices=30000,  # Set this to a reasonable maximum
        batch_size=2,
        num_joints=22
    )
    
    # Test the dataloader
    for vertices, weights, mask, triangles, num_vertices in dataloader:
        print(f"Batch vertices shape: {vertices.shape}")
        print(f"Batch weights shape: {weights.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Triangles shape: {triangles.shape}")
        print(f"Number of actual vertices: {num_vertices}")
        
        # Demonstrate using only valid vertices
        valid_vertices = vertices[mask]
        valid_weights = weights[mask]
        print(f"Valid vertices shape: {valid_vertices.shape}")
        print(f"Valid weights shape: {valid_weights.shape}")
        
        # Get joint positions for the first mesh in batch
        joint_positions = dataset.get_joint_positions(0)
        print(f"Joint positions shape: {joint_positions.shape}")
        break