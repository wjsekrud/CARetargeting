import numpy as np

def create_rotation_matrix(euler_angles):
    """
    오일러 각도(X,Y,Z 순서)로부터 3x3 회전 행렬을 생성합니다.
    
    Args:
        euler_angles: np.ndarray - [rx, ry, rz] 형태의 오일러 각도 (라디안)
    Returns:
        np.ndarray - 3x3 회전 행렬
    """
    rx, ry, rz = euler_angles
    
    # X축 회전
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    # Y축 회전
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # Z축 회전
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # 회전 행렬 결합 (Z * Y * X 순서)
    return Rz @ Ry @ Rx

def compute_bone_positions(joint_rotations, bone_offsets, parent_indices):
    """
    FK를 적용하여 각 본의 최종 위치를 계산합니다.
    
    Args:
        joint_rotations: np.ndarray - 각 조인트의 회전값 [num_joints, 3] (라디안)
        bone_offsets: np.ndarray - 각 본의 오프셋 [num_joints, 3]
        parent_indices: list - 각 본의 부모 본 인덱스 리스트. 루트는 -1
        
    Returns:
        np.ndarray - 각 본의 최종 위치 [num_joints, 3]
    """
    num_joints = len(bone_offsets)
    global_positions = np.zeros((num_joints, 3))  # 수정: float64 타입으로 명시적 생성
    global_rotations = np.zeros((num_joints, 3, 3))
    
    # 각 본에 대해 FK 적용
    for i in range(num_joints):
        parent_idx = parent_indices[i]
        local_rotation = create_rotation_matrix(joint_rotations[i])
        
        if parent_idx == -1:
            # 루트 본
            global_rotations[i] = local_rotation
            global_positions[i] = bone_offsets[i].copy()  # 수정: copy() 사용
        else:
            # 부모 본의 회전과 결합
            global_rotations[i] = global_rotations[parent_idx] @ local_rotation
            
            # 부모 본의 위치에서 현재 본의 오프셋을 회전시켜 더함
            rotated_offset = global_rotations[parent_idx] @ bone_offsets[i]
            # 수정: 명시적으로 배열 인덱싱을 사용하여 할당
            global_positions[i, :] = global_positions[parent_idx, :] + rotated_offset
            
    return global_positions

def example_usage():
    # 테스트 데이터
    num_joints = 3
    
    # 각 본의 회전값 (라디안)
    joint_rotations = np.array([
        [0, 0, np.pi/4],           # 루트 본: Z축으로 45도 회전
        [0, np.pi/6, 0],           # 첫 번째 자식: Y축으로 30도 회전
        [np.pi/6, 0, 0]            # 두 번째 자식: X축으로 30도 회전
    ])
    
    # 각 본의 오프셋
    bone_offsets = np.array([
        [0, 0, 0],                 # 루트 본
        [1, 0, 0],                 # 첫 번째 자식: X축 방향으로 1 단위
        [1, 0, 0]                  # 두 번째 자식: X축 방향으로 1 단위
    ])
    
    # 부모 본 인덱스 (-1은 루트를 의미)
    parent_indices = [-1, 0, 1]
    
    # FK 계산
    final_positions = compute_bone_positions(joint_rotations, bone_offsets, parent_indices)
    
    print("\nFinal bone positions:")
    for i, pos in enumerate(final_positions):
        print(f"Bone {i}: {pos}")
        
    # 디버깅을 위한 중간 값들 출력
    print("\nRotated offsets for verification:")
    for i in range(num_joints):
        if parent_indices[i] != -1:
            parent_rotation = create_rotation_matrix(joint_rotations[parent_indices[i]])
            rotated_offset = parent_rotation @ bone_offsets[i]
            print(f"Bone {i} rotated offset: {rotated_offset}")

#if __name__ == "__main__":
#    example_usage()