import numpy as np
from scipy import sparse

def convert_weights_to_sparse(weight_data, num_vertices, num_bones):
    """딕셔너리 형태의 가중치 데이터를 희소 행렬로 변환"""
    rows = []
    cols = []
    data = []
    
    for vertex_idx, bone_weights in enumerate(weight_data):
        for weight_info in bone_weights:
            rows.append(vertex_idx)
            cols.append(weight_info['bone_index'])
            data.append(weight_info['weight'])
    
    return sparse.csr_matrix((data, (rows, cols)), 
                           shape=(num_vertices, num_bones))

def compute_skinned_vertices_fast(weight_matrix, bone_transforms, vertices):
    """
    최적화된 LBS 구현
    
    Args:
        weight_matrix: scipy.sparse.csr_matrix - 버텍스-본 가중치 행렬 (N x B)
        bone_transforms: numpy.ndarray - 본 변환 행렬들 (B x 3)
        vertices: numpy.ndarray - 원본 버텍스 위치 (N x 3)
        
    Returns:
        numpy.ndarray - 스키닝된 버텍스 위치 (N x 3)
    """
    # 각 본의 변환을 적용한 버텍스 위치 계산 (브로드캐스팅 사용)
    transformed_positions = vertices[:, np.newaxis, :] + bone_transforms[np.newaxis, :, :]
    
    # 변환된 위치를 (N, B, 3) 형태로 재구성
    transformed_positions = transformed_positions.reshape(-1, bone_transforms.shape[0], 3)
    
    # 가중치 행렬과 변환된 위치를 곱하여 최종 위치 계산
    # weight_matrix: (N x B), transformed_positions: (N x B x 3)
    skinned_positions = np.zeros_like(vertices)
    
    for i in range(3):  # x, y, z 각 좌표에 대해
        skinned_positions[:, i] = weight_matrix.dot(transformed_positions[:, :, i])
    
    return skinned_positions

def example_usage_fast():
    # 테스트 데이터
    num_vertices = 1000
    num_bones = 22
    
    # 원본 가중치 데이터 (희소 형태)
    weight_data = [
        [{'idx': i % num_bones, 'weight': 0.7}, 
         {'idx': (i + 1) % num_bones, 'weight': 0.3}]
        for i in range(num_vertices)
    ]
    
    # 테스트용 데이터 생성
    vertices = np.random.rand(num_vertices, 3)
    bone_transforms = np.random.rand(num_bones, 3)
    
    # 가중치 데이터를 희소 행렬로 변환
    weight_matrix = convert_weights_to_sparse(weight_data, num_vertices, num_bones)
    
    # 스키닝 계산
    import time
    start_time = time.time()
    
    result = compute_skinned_vertices_fast(weight_matrix, bone_transforms, vertices)
    
    end_time = time.time()
    print(f"Processing time: {(end_time - start_time) * 1000:.2f}ms")
    print(f"Result shape: {result.shape}")

if __name__ == "__main__":
    example_usage_fast()