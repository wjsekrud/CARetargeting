import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class AABB:
    min_bound: torch.Tensor  # [..., 3]
    max_bound: torch.Tensor  # [..., 3]
    
    def intersects_batch(self, other: 'AABB') -> torch.Tensor:
        """배치 방식으로 AABB 교차 검사"""
        return torch.all(self.min_bound <= other.max_bound[..., None, :], dim=-1) & \
               torch.all(other.min_bound <= self.max_bound[..., None, :], dim=-1)

def compute_triangle_aabbs_batch(vertices: torch.Tensor, triangles: torch.Tensor) -> AABB:
    """배치 방식으로 모든 삼각형의 AABB를 한 번에 계산"""
    triangle_vertices = vertices[triangles]  # [N, 3, 3]
    min_bounds = torch.min(triangle_vertices, dim=1)[0]  # [N, 3]
    max_bounds = torch.max(triangle_vertices, dim=1)[0]  # [N, 3]
    return AABB(min_bounds, max_bounds)

def check_triangle_adjacency(triangles: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """삼각형 쌍이 서로 인접(버텍스를 공유)하는지 배치 방식으로 검사

    Args:
        triangles: [N, 3] 삼각형 인덱스 텐서
        pairs: [M, 2] 검사할 삼각형 쌍의 인덱스 텐서

    Returns:
        [M] 각 쌍의 인접 여부 (True: 인접함)
    """
    tri1 = triangles[pairs[:, 0]]  # [M, 3]
    tri2 = triangles[pairs[:, 1]]  # [M, 3]
    
    # 각 삼각형을 [M, 1, 3] 형태로 확장하여 브로드캐스팅
    tri1_expanded = tri1.unsqueeze(1)  # [M, 1, 3]
    tri2_expanded = tri2.unsqueeze(2)  # [M, 3, 1]
    
    # 버텍스 인덱스 비교를 통해 공유 버텍스 검출
    # eq 연산 후 any를 통해 각 버텍스가 다른 삼각형의 어떤 버텍스와 같은지 확인
    shared_vertices = (tri1_expanded == tri2_expanded).any(dim=2).any(dim=1)  # [M]
    
    return shared_vertices

def build_parallel_bvh(vertices: torch.Tensor, triangles: torch.Tensor, max_depth: int = 20):
    """병렬화된 BVH 구축
    
    각 레벨의 노드들을 배치로 처리하여 GPU 병렬성을 활용
    """
    num_triangles = triangles.shape[0]
    
    # 초기 AABB들을 한 번에 계산
    aabbs = compute_triangle_aabbs_batch(vertices, triangles)
    
    # 각 레벨별 노드 정보를 저장할 리스트
    level_nodes = [{
        'aabbs': aabbs,
        'triangle_indices': torch.arange(num_triangles, device=vertices.device),
        'is_leaf': torch.ones(num_triangles, dtype=torch.bool, device=vertices.device)
    }]
    
    for depth in range(max_depth):
        current_level = level_nodes[-1]
        if torch.all(current_level['is_leaf']):
            break
            
        # 중심점 계산을 배치로 수행
        tri_vertices = vertices[triangles[current_level['triangle_indices']]]
        centroids = torch.mean(tri_vertices, dim=1)
        
        # 분할 축 결정을 배치로 수행
        extent = torch.max(centroids, dim=0)[0] - torch.min(centroids, dim=0)[0]
        split_axis = torch.argmax(extent)
        
        # 중심점 기준 정렬을 배치로 수행
        centroid_values = centroids[:, split_axis]
        sorted_indices = torch.argsort(centroid_values)
        
        # 분할 위치 계산
        mid = len(sorted_indices) // 2
        
        # 자식 노드들의 정보 업데이트
        left_indices = current_level['triangle_indices'][sorted_indices[:mid]]
        right_indices = current_level['triangle_indices'][sorted_indices[mid:]]
        
        # 새로운 레벨의 노드 정보 생성
        next_level = {
            'aabbs': compute_triangle_aabbs_batch(vertices, triangles[torch.cat([left_indices, right_indices])]),
            'triangle_indices': torch.cat([left_indices, right_indices]),
            'is_leaf': torch.cat([
                torch.ones(mid, dtype=torch.bool, device=vertices.device),
                torch.ones(len(sorted_indices) - mid, dtype=torch.bool, device=vertices.device)
            ])
        }
        
        level_nodes.append(next_level)
    
    return level_nodes

def parallel_triangle_intersection(vertices: torch.Tensor, 
                                triangles: torch.Tensor,
                                pairs: torch.Tensor) -> torch.Tensor:
    """배치 방식으로 여러 삼각형 쌍의 교차 검사를 동시에 수행"""
    # 삼각형 쌍의 꼭지점 추출
    tri1_idx = pairs[:, 0]
    tri2_idx = pairs[:, 1]
    
    v1 = vertices[triangles[tri1_idx]]  # [N, 3, 3]
    v2 = vertices[triangles[tri2_idx]]  # [N, 3, 3]
    
    # 에지 벡터와 법선 계산을 배치로 수행
    e1 = v1[:, 1] - v1[:, 0]  # [N, 3]
    e2 = v1[:, 2] - v1[:, 0]  # [N, 3]
    n1 = torch.cross(e1, e2, dim=1)  # [N, 3]
    
    f1 = v2[:, 1] - v2[:, 0]  # [N, 3]
    f2 = v2[:, 2] - v2[:, 0]  # [N, 3]
    n2 = torch.cross(f1, f2, dim=1)  # [N, 3]
    
    # 분리축 후보들을 배치로 계산
    axes = torch.stack([
        n1, n2,
        torch.cross(e1, f1, dim=1),
        torch.cross(e1, f2, dim=1),
        torch.cross(e2, f1, dim=1),
        torch.cross(e2, f2, dim=1)
    ], dim=1)  # [N, 6, 3]
    
    # 각 축에 대한 투영 구간을 배치로 계산
    intersecting = torch.ones(pairs.shape[0], dtype=torch.bool, device=vertices.device)
    
    for i in range(6):
        axis = axes[:, i]  # [N, 3]
        axis_norm = torch.norm(axis, dim=1)
        valid_axis = axis_norm > 1e-10
        
        if not torch.any(valid_axis):
            continue
            
        # 투영 계산을 배치로 수행
        p1 = torch.bmm(v1, axis.unsqueeze(2)).squeeze(-1)  # [N, 3]
        p2 = torch.bmm(v2, axis.unsqueeze(2)).squeeze(-1)  # [N, 3]
        
        min1, _ = torch.min(p1, dim=1)  # [N]
        max1, _ = torch.max(p1, dim=1)  # [N]
        min2, _ = torch.min(p2, dim=1)  # [N]
        max2, _ = torch.max(p2, dim=1)  # [N]
        
        # 분리축 테스트를 배치로 수행
        intersecting &= ((max1 >= min2) & (max2 >= min1)) | ~valid_axis
    
    return intersecting

def find_collisions_parallel(vertices: torch.Tensor,
                           triangles: torch.Tensor,
                           level_nodes: List[dict]) -> List[Tuple[int, int]]:
    """병렬화된 충돌 검출"""
    device = vertices.device
    num_triangles = triangles.shape[0]
    
    # 모든 가능한 삼각형 쌍 생성
    tri_indices = torch.arange(num_triangles, device=device)
    pairs = torch.combinations(tri_indices, r=2)  # [N*(N-1)/2, 2]
    
    # AABB 교차 테스트를 배치로 수행
    aabbs = level_nodes[0]['aabbs']
    aabb_intersecting = aabbs.intersects_batch(aabbs)  # [N, N]
    
    # 상삼각 행렬만 사용
    aabb_intersecting = torch.triu(aabb_intersecting, diagonal=1)
    potential_collisions = torch.nonzero(aabb_intersecting)
    
    if len(potential_collisions) == 0:
        return []
    
    # 인접한 삼각형 제외
    non_adjacent = ~check_triangle_adjacency(triangles, potential_collisions)
    potential_collisions = potential_collisions[non_adjacent]
    
    if len(potential_collisions) == 0:
        return []
    
    # 실제 삼각형 교차 테스트를 배치로 수행
    intersecting = parallel_triangle_intersection(vertices, triangles, potential_collisions)
    collision_pairs = potential_collisions[intersecting]
    
    return [(int(i), int(j)) for i, j in collision_pairs.cpu().numpy()]

def detect_mesh_collisions_parallel(vertices: torch.Tensor, triangles: torch.Tensor):
    """병렬화된 메시 충돌 검출"""
    # 모든 텐서를 GPU로 이동
    vertices = vertices.cuda()
    triangles = triangles.cuda()
    
    # 병렬 BVH 구축
    level_nodes = build_parallel_bvh(vertices, triangles)
    
    # 병렬 충돌 검출
    return find_collisions_parallel(vertices, triangles, level_nodes)

# 테스트
if __name__ == "__main__":
    # 테스트용 메시 데이터
    vertices = torch.tensor([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [1., 1., 1.],
        [0., 0., -0.3]
    ], dtype=torch.float32)
    
    triangles = torch.tensor([
        [0, 1, 2],
        [1, 2, 3],
    ], dtype=torch.long)
    
    # 충돌 검출
    collisions = detect_mesh_collisions_parallel(vertices, triangles)
    print(f"충돌하는 삼각형 쌍: {collisions}")