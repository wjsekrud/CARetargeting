import torch
import numpy as np
from typing import List, Tuple, Set
import heapq

def compute_edge_lengths(vertices: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """삼각형의 각 엣지 길이를 계산합니다."""
    # 삼각형의 세 정점
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    
    # 각 엣지의 길이 계산
    e0 = torch.norm(v1 - v2, dim=1)  # edge opposite to v0
    e1 = torch.norm(v2 - v0, dim=1)  # edge opposite to v1
    e2 = torch.norm(v0 - v1, dim=1)  # edge opposite to v2
    
    return torch.stack([e0, e1, e2], dim=1)

def build_vertex_adjacency(vertices: torch.Tensor, triangles: torch.Tensor) -> List[Set[int]]:
    """각 정점에 인접한 정점들의 집합을 구축합니다."""
    num_vertices = vertices.shape[0]
    adjacency = [set() for _ in range(num_vertices)]
    
    # 삼각형의 각 엣지에 대해
    edges = torch.cat([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]]
    ])
    
    # 양방향 인접성 추가
    for v1, v2 in edges.cpu().numpy():
        adjacency[v1].add(v2)
        adjacency[v2].add(v1)
    
    return adjacency

def compute_vertex_triangles(vertices: torch.Tensor, triangles: torch.Tensor) -> List[Set[int]]:
    """각 정점이 속한 삼각형들의 집합을 구축합니다."""
    num_vertices = vertices.shape[0]
    vertex_triangles = [set() for _ in range(num_vertices)]
    
    for i, tri in enumerate(triangles):
        for v in tri:
            vertex_triangles[v.item()].add(i)
    
    return vertex_triangles

def update_distance(u: int, v: int, 
                   distances: torch.Tensor,
                   vertices: torch.Tensor,
                   triangles: torch.Tensor,
                   vertex_triangles: List[Set[int]],
                   edge_lengths: torch.Tensor) -> float:
    """두 정점 사이의 거리를 업데이트합니다."""
    # u와 v가 공유하는 삼각형 찾기
    common_triangles = vertex_triangles[u].intersection(vertex_triangles[v])
    
    min_dist = float('inf')
    for tri_idx in common_triangles:
        tri = triangles[tri_idx]
        
        # 삼각형에서 u와 v의 로컬 인덱스 찾기
        u_local = (tri == u).nonzero().item()
        v_local = (tri == v).nonzero().item()
        
        # 삼각형의 세 엣지 길이
        a, b, c = edge_lengths[tri_idx]
        
        # u와 v 사이의 직선 거리
        direct_dist = torch.norm(vertices[u] - vertices[v]).item()
        
        # 삼각형을 통과하는 거리 계산
        dist = direct_dist
        min_dist = min(min_dist, dist)
    
    return min_dist

def compute_geodesic_distance(vertices: torch.Tensor,
                            triangles: torch.Tensor,
                            start_vertex: int,
                            end_vertex: int) -> float:
    """Fast Marching Method를 사용하여 두 정점 사이의 geodesic distance를 계산합니다."""
    device = vertices.device
    num_vertices = vertices.shape[0]
    
    # 각 정점까지의 거리를 저장하는 배열
    distances = torch.full((num_vertices,), float('inf'), device=device)
    distances[start_vertex] = 0.0
    
    # 방문 상태를 추적
    visited = torch.zeros(num_vertices, dtype=torch.bool, device=device)
    
    # 전처리: 인접성 정보와 엣지 길이 계산
    adjacency = build_vertex_adjacency(vertices, triangles)
    vertex_triangles = compute_vertex_triangles(vertices, triangles)
    edge_lengths = compute_edge_lengths(vertices, triangles)
    
    # 우선순위 큐 초기화
    pq = [(0.0, start_vertex)]
    
    # Fast Marching
    while pq and not visited[end_vertex]:
        dist, current = heapq.heappop(pq)
        
        if visited[current]:
            continue
            
        visited[current] = True
        distances[current] = dist
        
        # 인접한 정점들 업데이트
        for neighbor in adjacency[current]:
            if visited[neighbor]:
                continue
                
            new_dist = distances[current] + update_distance(
                current, neighbor, distances, vertices, triangles, 
                vertex_triangles, edge_lengths
            )
            
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    
    return distances[end_vertex].item()

# Fast Marching Method를 배치로 처리하는 버전
def compute_geodesic_distances_batch(vertices: torch.Tensor,
                                   triangles: torch.Tensor,
                                   start_vertices: torch.Tensor,
                                   end_vertices: torch.Tensor) -> torch.Tensor:
    """여러 쌍의 정점들 사이의 geodesic distance를 병렬로 계산합니다."""
    num_pairs = start_vertices.shape[0]
    distances = torch.zeros(num_pairs, device=vertices.device)
    
    # 각 쌍에 대해 병렬로 계산
    for i in range(num_pairs):
        distances[i] = compute_geodesic_distance(
            vertices, triangles,
            start_vertices[i].item(),
            end_vertices[i].item()
        )
    
    return distances

# 사용 예시:
if __name__ == "__main__":
    # 테스트용 메시 데이터
    vertices = torch.tensor([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [1., 1., 0.],
        [0.5, 0.5, 1.]
    ], dtype=torch.float32)
    
    triangles = torch.tensor([
        [0, 1, 2],
        [1, 3, 2],
        [1, 4, 3],
        [2, 3, 4]
    ], dtype=torch.long)
    
    # 두 정점 사이의 geodesic distance 계산
    start_vertex = 0
    end_vertex = 4
    
    distance = compute_geodesic_distance(vertices, triangles, start_vertex, end_vertex)
    print(f"Geodesic distance between vertices {start_vertex} and {end_vertex}: {distance:.4f}")
    
    # 여러 쌍의 정점들에 대해 배치로 계산
    start_vertices = torch.tensor([0, 1, 2])
    end_vertices = torch.tensor([4, 3, 4])
    
    distances = compute_geodesic_distances_batch(vertices, triangles, start_vertices, end_vertices)
    print("Batch geodesic distances:", distances)