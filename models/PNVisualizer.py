import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PointNet import MeshAutoEncoder, compute_parent_joint_offsets
from PNdataloader import create_mesh_dataloader

class ReconstructionVisualizer:
    def __init__(self):
        # Create a custom colormap: blue (low error) to red (high error)
        self.error_cmap = LinearSegmentedColormap.from_list('error_map', [
            (0, 'blue'),      # Low error
            (0.5, 'green'),   # Medium error
            (1, 'red')        # High error
        ])
    
    def compute_vertex_errors(self, original_vertices: torch.Tensor, 
                            reconstructed_vertices: torch.Tensor,
                            mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute per-vertex reconstruction error
        
        Args:
            original_vertices: [B, N, 3] original vertex positions
            reconstructed_vertices: [B, N, 3] reconstructed vertex positions
            mask: [B, N] boolean mask for valid vertices
            
        Returns:
            [B, N] tensor of per-vertex errors
        """
        # Compute Euclidean distance between original and reconstructed vertices
        errors = torch.norm(original_vertices - reconstructed_vertices, dim=-1)
        
        if mask is not None:
            # Set error to 0 for invalid vertices
            errors = errors * mask.float()
        
        return errors
    
    def errors_to_colors(self, errors: torch.Tensor, 
                        mask: torch.Tensor = None) -> np.ndarray:
        """
        Convert error values to RGB colors
        
        Args:
            errors: [N] tensor of error values
            mask: [N] boolean mask for valid vertices
            
        Returns:
            [N, 3] array of RGB colors
        """
        errors_np = errors.detach().cpu().numpy()
        
        if mask is not None:
            mask_np = mask.detach().cpu().numpy()
            # Normalize errors only considering valid vertices
            valid_errors = errors_np[mask_np]
            if len(valid_errors) > 0:
                error_min = valid_errors.min()
                error_max = valid_errors.max()
                normalized_errors = (errors_np - error_min) / (error_max - error_min)
            else:
                normalized_errors = np.zeros_like(errors_np)
        else:
            # Normalize all errors
            error_min = errors_np.min()
            error_max = errors_np.max()
            normalized_errors = (errors_np - error_min) / (error_max - error_min)
        
        # Convert to colors using colormap
        colors = self.error_cmap(normalized_errors)[:, :3]
        
        return colors
    
    def save_error_visualization(self, vertices: np.ndarray, 
                               colors: np.ndarray,
                               triangles: np.ndarray,
                               output_path: str,
                               error_stats: dict = None):
        """
        Save mesh with error visualization as PLY file
        
        Args:
            vertices: [N, 3] array of vertex positions
            colors: [N, 3] array of RGB colors representing errors
            triangles: [M, 3] array of triangle indices
            output_path: Path to save the PLY file
            error_stats: Optional dictionary with error statistics
        """
        # Convert colors to 8-bit RGB
        colors_uint8 = (colors * 255).astype(np.uint8)
        
        with open(output_path, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element face {len(triangles)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices with colors
            for vertex, color in zip(vertices, colors_uint8):
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} "
                       f"{color[0]} {color[1]} {color[2]}\n")
            
            # Write triangle faces
            for triangle in triangles:
                f.write(f"3 {triangle[0]} {triangle[1]} {triangle[2]}\n")
        
        # Save error statistics if provided
        if error_stats is not None:
            stats_path = Path(output_path).with_suffix('.txt')
            with open(stats_path, 'w') as f:
                for key, value in error_stats.items():
                    f.write(f"{key}: {value}\n")

def visualize_reconstruction(model_path: str, mesh_path: str, output_dir: str):
    """
    Load model and mesh, compute reconstruction, and visualize errors
    
    Args:
        model_path: Path to saved model checkpoint
        mesh_path: Path to input mesh file
        output_dir: Directory to save visualization results
    """
    # Load full model (need both encoder and decoder for reconstruction)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = MeshAutoEncoder(num_joints=checkpoint['num_joints'],
                           feature_dim=checkpoint['feature_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    # Load mesh data
    loader, dataset = create_mesh_dataloader(
        mesh_dir="./dataset/vertexes",
        max_vertices=21000,
        batch_size=1,
        num_joints=22,
        file_pattern='*_clean_vertices.txt'
    )
    for vertices_positions, skinning_weights, vertmask, triangles, num_vertices in loader:
        vertices = vertices_positions.to(device)
        weights = skinning_weights.to(device)
        if vertmask is not None:
            mask = vertmask.to(device)
        joint_positions = []
        for i in range(vertices.size(0)):
            joints = dataset.get_joint_positions(i)
            joint_positions.append(torch.FloatTensor(joints))
        joint_positions = torch.stack(joint_positions).to(device)
        parent_offsets, _ = compute_parent_joint_offsets(vertices, skinning_weights, joint_positions)
        break
        
    # Compute reconstruction
    with torch.no_grad():
        features, reconstructed_vertices = model(vertices, weights, parent_offsets)
    
    # Create visualizer and compute errors
    visualizer = ReconstructionVisualizer()
    errors = visualizer.compute_vertex_errors(vertices, reconstructed_vertices, mask)
    
    # Compute error statistics
    if mask is not None:
        valid_errors = errors[mask]
    else:
        valid_errors = errors
        
    error_stats = {
        'mean_error': valid_errors.mean().item(),
        'max_error': valid_errors.max().item(),
        'min_error': valid_errors.min().item(),
        'std_error': valid_errors.std().item()
    }
    
    # Convert errors to colors and save visualization
    colors = visualizer.errors_to_colors(errors.squeeze(0), 
                                       mask.squeeze(0) if mask is not None else None)
    
    output_path = Path(output_dir) / 'reconstruction_error.ply'
    visualizer.save_error_visualization(
        vertices=vertices.squeeze(0).cpu().numpy(),
        colors=colors,
        triangles=triangles,
        output_path=str(output_path),
        error_stats=error_stats
    )
    
    # Save reconstructed mesh as well
    output_recon_path = Path(output_dir) / 'reconstructed_mesh.obj'
    save_to_obj(reconstructed_vertices.squeeze(0).cpu().numpy(),
             triangles,
             output_recon_path)
    
    print(f"Error Statistics:")
    for key, value in error_stats.items():
        print(f"{key}: {value:.6f}")

def save_to_obj(vertices, tris, filepath="C:/Users/vml/Documents/GitHub/CARetargeting/dataset/output.obj"):
    """
    스키닝된 버텍스와 폴리곤 정보를 OBJ 파일로 저장합니다.
    
    Args:
        vertices: numpy.ndarray - 버텍스 위치 배열 (N x 3)
        tris: List[List[int]] - 삼각형을 구성하는 버텍스 인덱스 리스트 (CW 순서)
        filepath: str - 저장할 OBJ 파일 경로
    """
    print(vertices.shape)
    with open(filepath, 'w') as f:
        # 파일 헤더 작성
        f.write("# Exported by LBS Skinning Validator\n")
        f.write("# Vertices: {}\n".format(len(vertices)))
        f.write("# Faces: {}\n\n".format(len(tris)))
        #self.clip.append(vertex_positions)
        
        # 버텍스 정보 작성
        for vertex in vertices:
            #print(vertex)
            #print(vertex)
            # OBJ 파일은 Y-up 좌표계를 사용하므로, 필요한 경우 여기서 좌표계 변환을 수행할 수 있습니다
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(vertex[0], vertex[1], vertex[2]))
        
        f.write("\n")

         # 폴리곤 정보 작성
        # OBJ 파일의 인덱스는 1부터 시작하므로 1을 더해줍니다
        for tri in tris:
            #print(tri)
            # CW -> CCW 변환 (필요한 경우)
            # f.write("f {} {} {}\n".format(tri[0]+1, tri[2]+1, tri[1]+1))
            # CW 유지
            f.write("f {} {} {}\n".format(tri[0]+1, tri[1]+1, tri[2]+1))

# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize reconstruction errors')
    parser.add_argument('--model_path', type=str, default="./dataset/PointNet/full_model_best.pth",
                       help='Path to saved model checkpoint')
    parser.add_argument('--mesh_path', type=str, default="./",
                       help='Path to input mesh file')
    parser.add_argument('--output_dir', type=str, default="./dataset/PointNet",
                       help='Directory to save visualization results')
    
    args = parser.parse_args()
    visualize_reconstruction(args.model_path, args.mesh_path, args.output_dir)