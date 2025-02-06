import numpy as np
from aPyOpenGL.transforms.numpy import quat

def create_rotation_matrix_from_quaternion(quaternion):
    """
    쿼터니언으로부터 3x3 회전 행렬을 생성합니다.
    
    Args:
        quaternion: np.ndarray - [w, x, y, z] 형태의 쿼터니언
    Returns:
        np.ndarray - 3x3 회전 행렬
    """
    w, x, y, z = quaternion
    
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def create_local_coordinate_frame(up_vector):
    """
    주어진 up 벡터로부터 로컬 좌표계를 생성합니다.
    
    Args:
        up_vector: np.ndarray - 조인트의 up 벡터 [3,]
    Returns:
        np.ndarray - 3x3 좌표계 변환 행렬 (각 행은 local x, y, z 축)
    """
    # up 벡터를 정규화
    y_axis = up_vector / np.linalg.norm(up_vector)
    
    # up 벡터와 수직인 임시 벡터 생성 (z축)
    if np.allclose(y_axis, [0, 1, 0]):
        z_axis = np.array([0, 0, 1])
    else:
        z_axis = np.cross([0, 1, 0], y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
    
    # x축은 y와 z의 외적
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # 좌표계 변환 행렬 생성
    return np.vstack([x_axis, y_axis, z_axis])

def compute_bone_positions(joint_rotations, bone_offsets, parent_indices, joint_orientations):
    """
    FK를 적용하여 각 본의 최종 위치를 계산합니다.
    
    Args:
        joint_rotations: np.ndarray - 각 조인트의 쿼터니언 회전값 [num_joints, 4] (w,x,y,z)
        bone_offsets: np.ndarray - 각 본의 오프셋 [num_joints, 3]
        parent_indices: list - 각 본의 부모 본 인덱스 리스트. 루트는 -1
        joint_orientations: np.ndarray - 각 조인트의 up 벡터 [num_joints, 3]
        
    Returns:
        np.ndarray - 각 본의 최종 위치 [num_joints, 3]
        np.ndarray - 각 본의 최종 회전 행렬 [num_joints, 3, 3]
    """


    num_joints = len(bone_offsets)
    global_positions = np.zeros((num_joints, 3))
    global_rotations = np.zeros((num_joints, 3, 3))
    
    # 각 본에 대해 FK 적용
    for i in range(num_joints):
        local_frame = create_local_coordinate_frame(joint_orientations[i])
        local_rotation = create_rotation_matrix_from_quaternion(joint_rotations[i])
        
        # 로컬 프레임에서의 회전을 적용
        combined_rotation = local_frame.T @ local_rotation @ local_frame
        
        parent_idx = parent_indices[i]
        if parent_idx == -1:
            # 루트 본
            global_rotations[i] = combined_rotation
            global_positions[i] = bone_offsets[i].copy()
        else:
            # 부모 본의 회전과 결합
            global_rotations[i] = global_rotations[parent_idx] @ combined_rotation
            
            # 부모 본의 위치에서 현재 본의 오프셋을 회전시켜 더함
            rotated_offset = global_rotations[parent_idx] @ bone_offsets[i]
            global_positions[i] = global_positions[parent_idx] + rotated_offset
            
    return global_positions, global_rotations

def example_usage():
    # 테스트 데이터
    num_joints = 3
    
    # 각 본의 쿼터니언 회전값 (w,x,y,z)
    joint_rotations = np.array([
        [np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)],  # 루트 본: Z축으로 45도 회전
        [np.cos(np.pi/12), 0, np.sin(np.pi/12), 0], # 첫 번째 자식: Y축으로 30도 회전
        [np.cos(np.pi/12), np.sin(np.pi/12), 0, 0]  # 두 번째 자식: X축으로 30도 회전
    ])
    
    # 각 본의 오프셋
    bone_offsets = np.array([
        [0, 0, 0],     # 루트 본
        [1, 0, 0],     # 첫 번째 자식
        [1, 0, 0]      # 두 번째 자식
    ])
    
    # 각 조인트의 로컬 좌표계 방향 (up 벡터)
    joint_orientations = np.array([
        [0, 1, 0],     # 루트 본: Y축이 up
        [0, 1, 0],     # 첫 번째 자식: Y축이 up
        [1, 0, 0]      # 두 번째 자식: X축이 up (비틀어진 조인트)
    ])
    
    # 부모 본 인덱스
    parent_indices = [-1, 0, 1]
    
    # FK 계산
    positions, rotations = compute_bone_positions(
        joint_rotations, 
        bone_offsets, 
        parent_indices,
        joint_orientations
    )
    
    print("\nFinal bone positions:")
    for i, pos in enumerate(positions):
        print(f"Bone {i}: {pos}")
        
    print("\nFinal bone rotations (as matrices):")
    for i, rot in enumerate(rotations):
        print(f"Bone {i}:\n{rot}\n")