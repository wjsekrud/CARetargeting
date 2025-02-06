import numpy as np
from scipy import sparse

def create_transform_matrix(rotation, translation):
    """
    회전과 이동으로부터 4x4 변환 행렬을 생성합니다.
    
    Args:
        rotation: np.ndarray - 3x3 회전 행렬
        translation: np.ndarray - [x, y, z] 이동 벡터
    
    Returns:
        np.ndarray - 4x4 변환 행렬
    """
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix

def compute_bone_transforms(joint_rotations, bone_offsets, parent_indices):
    """
    각 본에 대한 글로벌 변환 행렬을 계산합니다.
    
    Args:
        joint_rotations: np.ndarray - [num_joints, 3, 3] 각 관절의 회전 행렬
        bone_offsets: np.ndarray - [num_joints, 3] 각 본의 오프셋
        parent_indices: list - 각 본의 부모 인덱스 (-1은 루트)
    
    Returns:
        np.ndarray - [num_joints, 4, 4] 각 본의 글로벌 변환 행렬
    """
    num_joints = len(bone_offsets)
    global_transforms = np.zeros((num_joints, 4, 4))
    
    # 각 본에 대해 순차적으로 처리
    for i in range(num_joints):
        local_transform = create_transform_matrix(joint_rotations[i], bone_offsets[i])
        
        if parent_indices[i] == -1:
            # 루트 본
            global_transforms[i] = local_transform
        else:
            # 부모 본의 변환을 현재 본에 적용
            parent_transform = global_transforms[parent_indices[i]]
            global_transforms[i] = parent_transform @ local_transform
    
    return global_transforms

def convert_weights_to_sparse(weight_data, num_vertices, num_bones):
    """
    가중치 데이터를 희소 행렬로 변환합니다.
    
    Args:
        weight_data: List[List[Dict]] - 각 버텍스의 본 가중치 정보
        num_vertices: int - 버텍스 수
        num_bones: int - 본 수
    
    Returns:
        scipy.sparse.csr_matrix - 희소 가중치 행렬
    """
    rows = []
    cols = []
    data = []
    
    for vertex_idx, bone_weights in enumerate(weight_data):
        for weight_info in bone_weights:
            rows.append(vertex_idx)
            cols.append(weight_info['idx'])
            data.append(weight_info['weight'])
    
    return sparse.csr_matrix((data, (rows, cols)), 
                           shape=(num_vertices, num_bones))

def compute_skinning_matrices(bone_transforms, inverse_bind_matrices):
    """
    스키닝 행렬을 계산합니다.
    
    Args:
        bone_transforms: np.ndarray - [num_joints, 4, 4] 각 본의 현재 변환 행렬
        inverse_bind_matrices: np.ndarray - [num_joints, 4, 4] 각 본의 역바인드 행렬
    
    Returns:
        np.ndarray - [num_joints, 4, 4] 각 본의 스키닝 행렬
    """
    return np.matmul(bone_transforms, inverse_bind_matrices)

def compute_skinned_vertices(vertices, weight_matrix, skinning_matrices):
    """
    LBS를 사용하여 버텍스 위치를 계산합니다.
    
    Args:
        vertices: np.ndarray - [num_vertices, 3] 원본 버텍스 위치
        weight_matrix: scipy.sparse.csr_matrix - [num_vertices, num_bones] 가중치 행렬
        skinning_matrices: np.ndarray - [num_bones, 4, 4] 스키닝 행렬
    
    Returns:
        np.ndarray - [num_vertices, 3] 스키닝된 버텍스 위치
    """
    num_vertices = len(vertices)
    
    # 동차 좌표계로 변환 (w=1 추가)
    vertices_homogeneous = np.ones((num_vertices, 4))
    vertices_homogeneous[:, :3] = vertices
    
    # 각 본의 영향을 계산
    transformed_vertices = np.zeros((num_vertices, 4))
    
    for i in range(skinning_matrices.shape[0]):
        # 현재 본에 대한 가중치
        weights = weight_matrix[:, i].toarray().flatten()
        
        # 0이 아닌 가중치를 가진 버텍스만 처리
        nonzero_indices = weights.nonzero()[0]
        if len(nonzero_indices) > 0:
            # 현재 본의 변환 적용
            bone_transform = skinning_matrices[i]
            transformed = vertices_homogeneous[nonzero_indices] @ bone_transform.T
            
            # 가중치를 적용하여 결과에 누적
            transformed_vertices[nonzero_indices] += transformed * weights[nonzero_indices, np.newaxis]
    
    # 동차 좌표에서 3D 좌표로 변환
    return transformed_vertices[:, :3]

def compute_inverse_bind_matrices(bind_rotations, bind_translations, parent_indices):
    """
    바인드 포즈의 역행렬을 계산합니다.
    
    Args:
        bind_rotations: np.ndarray - [num_joints, 3, 3] 바인드 포즈의 회전 행렬
        bind_translations: np.ndarray - [num_joints, 3] 바인드 포즈의 이동 벡터
        parent_indices: list - 각 본의 부모 인덱스
    
    Returns:
        np.ndarray - [num_joints, 4, 4] 역바인드 행렬
    """
    num_joints = len(parent_indices)
    bind_matrices = np.zeros((num_joints, 4, 4))
    
    # 바인드 포즈의 글로벌 변환 행렬 계산
    for i in range(num_joints):
        local_transform = create_transform_matrix(bind_rotations[i], bind_translations[i])
        
        if parent_indices[i] == -1:
            bind_matrices[i] = local_transform
        else:
            parent_transform = bind_matrices[parent_indices[i]]
            bind_matrices[i] = parent_transform @ local_transform
    
    # 역행렬 계산
    return np.linalg.inv(bind_matrices)

def example_usage():
    # 테스트 데이터 생성
    num_vertices = 4
    num_joints = 3
    
    # 버텍스 위치
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ])
    
    # 스키닝 가중치
    weight_data = [
        [{'idx': 0, 'weight': 0.7}, {'idx': 1, 'weight': 0.3}],
        [{'idx': 0, 'weight': 1.0}],
        [{'idx': 1, 'weight': 1.0}],
        [{'idx': 0, 'weight': 0.5}, {'idx': 1, 'weight': 0.5}]
    ]
    
    # 본 계층 구조
    parent_indices = [-1, 0, 1]
    
    # 바인드 포즈 데이터
    bind_rotations = np.array([np.eye(3) for _ in range(num_joints)])
    bind_translations = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 0]
    ])
    
    # 현재 포즈 데이터 (예: 각 본을 45도 회전)
    angle = np.pi / 4
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    joint_rotations = np.array([rotation_matrix for _ in range(num_joints)])
    
    # 1. 가중치 행렬 생성
    weight_matrix = convert_weights_to_sparse(weight_data, num_vertices, num_joints)
    
    # 2. 역바인드 행렬 계산
    inverse_bind_matrices = compute_inverse_bind_matrices(
        bind_rotations, bind_translations, parent_indices)
    
    # 3. 현재 본 변환 계산
    bone_transforms = compute_bone_transforms(
        joint_rotations, bind_translations, parent_indices)
    
    # 4. 스키닝 행렬 계산
    skinning_matrices = compute_skinning_matrices(
        bone_transforms, inverse_bind_matrices)
    
    # 5. 최종 버텍스 위치 계산
    result = compute_skinned_vertices(vertices, weight_matrix, skinning_matrices)
    
    print("Original vertices:")
    print(vertices)
    print("\nSkinned vertices:")
    print(result)

if __name__ == "__main__":
    example_usage()