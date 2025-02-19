import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class AABB:
    min_bound: torch.Tensor  # [..., 3]
    max_bound: torch.Tensor  # [..., 3]
    
    def intersects_batch(self, other: 'AABB') -> torch.Tensor:
        return torch.all(self.min_bound <= other.max_bound[..., None, :], dim=-1) & \
               torch.all(other.min_bound <= self.max_bound[..., None, :], dim=-1)
    
    def get_size(self) -> torch.Tensor:
        return self.max_bound - self.min_bound

def compute_triangle_aabbs_batch(vertices: torch.Tensor, triangles: torch.Tensor) -> AABB:
    triangle_vertices = vertices[triangles]  # [N, 3, 3]
    min_bounds = torch.min(triangle_vertices, dim=1)[0]  # [N, 3]
    max_bounds = torch.max(triangle_vertices, dim=1)[0]  # [N, 3]
    return AABB(min_bounds, max_bounds)

def compute_node_aabb(vertices: torch.Tensor, triangles: torch.Tensor, triangle_indices: torch.Tensor) -> AABB:
    """노드에 포함된 모든 삼각형을 감싸는 단일 AABB를 계산"""
    triangle_vertices = vertices[triangles[triangle_indices]]  # [N, 3, 3]
    min_bounds = torch.min(triangle_vertices.reshape(-1, 3), dim=0)[0]  # [3]
    max_bounds = torch.max(triangle_vertices.reshape(-1, 3), dim=0)[0]  # [3]
    return AABB(min_bounds, max_bounds)

@dataclass
class BVHNode:
    aabb: AABB
    triangle_indices: torch.Tensor
    left: Optional['BVHNode'] = None
    right: Optional['BVHNode'] = None
    
    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None
    
    def get_leaf_node_indices(self) -> List[torch.Tensor]:
        """이 노드 아래의 모든 리프 노드의 삼각형 인덱스들을 수집"""
        if self.is_leaf:
            return [self.triangle_indices]
        
        indices = []
        if self.left:
            indices.extend(self.left.get_leaf_node_indices())
        if self.right:
            indices.extend(self.right.get_leaf_node_indices())
        return indices


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
 
def build_bvh_recursive(vertices: torch.Tensor, triangles: torch.Tensor, 
                       triangle_indices: torch.Tensor, depth: int = 0, max_depth: int = 20, 
                       min_triangles: int = 4) -> BVHNode:
    """재귀적으로 BVH를 구축하는 함수"""
    # 노드의 AABB 계산 (모든 삼각형을 포함하는 단일 박스)
    node_aabb = compute_node_aabb(vertices, triangles, triangle_indices)
    node = BVHNode(aabb=node_aabb, triangle_indices=triangle_indices)
    
    # 종료 조건
    if len(triangle_indices) <= min_triangles or depth >= max_depth:
        return node
        
    # 삼각형의 중심점 계산
    tri_vertices = vertices[triangles[triangle_indices]]
    centroids = torch.mean(tri_vertices, dim=1)
    
    # 분할 축 결정 (가장 긴 축 선택)
    node_size = node_aabb.get_size()  # 이제 크기 [3]의 텐서
    split_axis = torch.argmax(node_size)
    
    # 중심점 기준 정렬 및 분할
    centroid_values = centroids[:, split_axis]
    sorted_indices = torch.argsort(centroid_values)
    mid = len(sorted_indices) // 2
    
    left_indices = triangle_indices[sorted_indices[:mid]]
    right_indices = triangle_indices[sorted_indices[mid:]]
    
    # 재귀적으로 자식 노드 구축
    node.left = build_bvh_recursive(vertices, triangles, left_indices, depth + 1, max_depth, min_triangles)
    node.right = build_bvh_recursive(vertices, triangles, right_indices, depth + 1, max_depth, min_triangles)

    return node

def parallel_triangle_intersection(vertices: torch.Tensor, 
                                triangles: torch.Tensor,
                                pairs: torch.Tensor,
                                epsilon: float = 1e-6) -> torch.Tensor:
    """개선된 삼각형 교차 검사"""
    tri1_idx = pairs[:, 0]
    tri2_idx = pairs[:, 1]
    
    v1 = vertices[triangles[tri1_idx]]  # [N, 3, 3]
    v2 = vertices[triangles[tri2_idx]]  # [N, 3, 3]
    
    e1 = v1[:, 1] - v1[:, 0]  # [N, 3]
    e2 = v1[:, 2] - v1[:, 0]  # [N, 3]
    n1 = torch.cross(e1, e2, dim=1)  # [N, 3]
    
    f1 = v2[:, 1] - v2[:, 0]  # [N, 3]
    f2 = v2[:, 2] - v2[:, 0]  # [N, 3]
    n2 = torch.cross(f1, f2, dim=1)  # [N, 3]
    
    # 법선 벡터 정규화
    n1_norm = torch.norm(n1, dim=1, keepdim=True)
    n2_norm = torch.norm(n2, dim=1, keepdim=True)
    valid_normals = (n1_norm > epsilon) & (n2_norm > epsilon)
    n1 = torch.where(valid_normals, n1 / n1_norm, n1)
    n2 = torch.where(valid_normals, n2 / n2_norm, n2)
    
    # 분리축 후보들 계산 및 정규화
    axes = []
    axes.append(n1)
    axes.append(n2)
    
    # 에지 크로스 프로덕트 계산 및 정규화
    for e in [e1, e2]:
        for f in [f1, f2]:
            cross = torch.cross(e, f, dim=1)
            cross_norm = torch.norm(cross, dim=1, keepdim=True)
            valid_cross = cross_norm > epsilon
            normalized_cross = torch.where(valid_cross, cross / cross_norm, cross)
            axes.append(normalized_cross)
    
    axes = torch.stack(axes, dim=1)  # [N, 6, 3]
    
    intersecting = torch.ones(pairs.shape[0], dtype=torch.bool, device=vertices.device)
    
    for i in range(axes.shape[1]):
        axis = axes[:, i]  # [N, 3]
        axis_norm = torch.norm(axis, dim=1)
        valid_axis = axis_norm > epsilon
        
        if not torch.any(valid_axis):
            continue
            
        p1 = torch.bmm(v1, axis.unsqueeze(2)).squeeze(-1)  # [N, 3]
        p2 = torch.bmm(v2, axis.unsqueeze(2)).squeeze(-1)  # [N, 3]
        
        min1, _ = torch.min(p1, dim=1)  # [N]
        max1, _ = torch.max(p1, dim=1)  # [N]
        min2, _ = torch.min(p2, dim=1)  # [N]
        max2, _ = torch.max(p2, dim=1)  # [N]
        
        gap = torch.min(max1 - min2, max2 - min1)
        intersecting &= ((gap >= -epsilon) | ~valid_axis)
    
    return intersecting

def find_collisions_parallel_batched(vertices: torch.Tensor,
                                   triangles: torch.Tensor,
                                   root: BVHNode,
                                   batch_size: int = 10000) -> List[Tuple[int, int]]:
    """계층적 BVH와 배치 처리를 결합한 충돌 검출"""
    print("startFCPB")

    # 리프 노드들의 인덱스 수집
    leaf_indices_lists = root.get_leaf_node_indices()
    

    # 리프 노드들의 AABB 계산
    leaf_aabbs = [compute_node_aabb(vertices, triangles, indices) for indices in leaf_indices_lists]
    
    # 모든 리프 노드들의 AABB를 하나의 텐서로 모음
    min_bounds = torch.stack([aabb.min_bound for aabb in leaf_aabbs])  # [N, 3]
    max_bounds = torch.stack([aabb.max_bound for aabb in leaf_aabbs])  # [N, 3]
    
    # 모든 AABB 쌍을 한 번에 비교
    all_aabb = AABB(min_bounds, max_bounds)
    intersecting_matrix = all_aabb.intersects_batch(all_aabb)  # [N, N]
    
    # 상삼각 행렬만 사용 (중복 제거)
    intersecting_matrix = torch.triu(intersecting_matrix, diagonal=1)
    intersecting_pairs = torch.nonzero(intersecting_matrix)  # [M, 2]
    
    # 메모리 효율적인 배치 처리
    final_collisions = []
    batch_size = 1000  # AABB 쌍 배치 크기
    
    print("startcreatingpairs, IPlen: ", len(intersecting_pairs))
    for batch_start in range(0, len(intersecting_pairs), batch_size):
        batch_end = min(batch_start + batch_size, len(intersecting_pairs))
        batch_pairs = intersecting_pairs[batch_start:batch_end]
        
        # 현재 배치의 삼각형 쌍 생성
        batch_collision_pairs = []
        for i, j in batch_pairs:
            pairs = torch.cartesian_prod(leaf_indices_lists[i], leaf_indices_lists[j])
            # 자기 자신과의 충돌은 즉시 제외
            valid_pairs = pairs[:, 0] != pairs[:, 1]
            pairs = pairs[valid_pairs]
            if len(pairs) > 0:
                batch_collision_pairs.append(pairs)
        
        if not batch_collision_pairs:
            continue
            
        # 배치 내의 모든 쌍 합치기
        all_pairs = torch.cat(batch_collision_pairs, dim=0)
    
    if len(all_pairs) == 0:
        return []
    
    # 인접한 삼각형 제외
    non_adjacent = ~check_triangle_adjacency(triangles, all_pairs)
    all_pairs = all_pairs[non_adjacent]
    
    if len(all_pairs) == 0:
        return []
    
    # 배치 크기로 나누어 처리
    print("startfindingintersections")
    final_collisions = []
    for i in range(0, len(all_pairs), batch_size):
        batch_pairs = all_pairs[i:i + batch_size]
        intersecting = parallel_triangle_intersection(vertices, triangles, batch_pairs)
        collision_batch = batch_pairs[intersecting]
        final_collisions.extend([(int(i), int(j)) for i, j in collision_batch.cpu().numpy()])
    
    return final_collisions

def detect_mesh_collisions_parallel(vertices: torch.Tensor, triangles: torch.Tensor):
    """병렬화된 메시 충돌 검출"""
    # 모든 텐서를 GPU로 이동
    vertices = vertices.cuda()
    triangles = triangles.cuda()
    
    # 초기 삼각형 인덱스
    triangle_indices = torch.arange(triangles.shape[0], device=vertices.device)
    
    # BVH 구축 (계층적)
    root = build_bvh_recursive(vertices, triangles, triangle_indices)
    
    # 병렬 배치 처리로 충돌 검출
    return find_collisions_parallel_batched(vertices, triangles, root)

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