import numpy as np

class Joint:
    def __init__(self, name, offset, channels=None, children=None):
        self.name = name
        self.offset = offset
        self.channels = channels if channels else ["Xrotation", "Yrotation", "Zrotation"]
        self.children = children if children else []

def create_skeleton_hierarchy(parent_indices, joint_offsets, joint_names=None):
    """스켈레톤 계층 구조 생성"""
    num_joints = len(parent_indices)
    if joint_names is None:
        joint_names = [f"joint_{i}" for i in range(num_joints)]
    
    # 각 조인트의 자식 리스트 생성
    children_dict = {i: [] for i in range(num_joints)}
    for i, parent in enumerate(parent_indices):
        if parent != -1:
            children_dict[parent].append(i)
    
    # Joint 객체 생성
    joints = {}
    for i in range(num_joints):
        joints[i] = Joint(
            name=joint_names[i],
            offset=joint_offsets[i],
            children=[]
        )
    
    # 자식 연결
    for i, children in children_dict.items():
        joints[i].children = [joints[child] for child in children]
    
    # 루트 조인트 찾기
    root = joints[parent_indices.index(-1)]
    return root

def write_joint_hierarchy(f, joint, level=0):
    """계층 구조를 BVH 형식으로 작성"""
    indent = "  " * level
    
    if level == 0:
        f.write(f"{indent}HIERARCHY\n")
        f.write(f"{indent}ROOT {joint.name}\n")
    
    f.write(f"{indent}{{\n")
    f.write(f"{indent}  OFFSET {joint.offset[0]:.6f} {joint.offset[1]:.6f} {joint.offset[2]:.6f}\n")
    
    if joint.children:
        f.write(f"{indent}  CHANNELS {len(joint.channels)}")
        for channel in joint.channels:
            f.write(f" {channel}")
        f.write("\n")
        
        for child in joint.children:
            f.write(f"{indent}  JOINT {child.name}\n")
            write_joint_hierarchy(f, child, level + 1)
    else:
        f.write(f"{indent}  End Site\n")
        f.write(f"{indent}  {{\n")
        f.write(f"{indent}    OFFSET 0.0 0.0 0.0\n")
        f.write(f"{indent}  }}\n")
    
    f.write(f"{indent}}}\n")

def write_motion_data(f, animation_data, frame_time=0.0333333):
    """모션 데이터를 BVH 형식으로 작성"""
    num_frames, num_joints, _ = animation_data.shape
    
    f.write("\nMOTION\n")
    f.write(f"Frames: {num_frames}\n")
    f.write(f"Frame Time: {frame_time:.7f}\n")
    
    for frame in range(num_frames):
        frame_data = []
        for joint in range(num_joints):
            # XYZ 순서로 회전값 추가
            rotations = animation_data[frame, joint]
            # 라디안을 도(degree)로 변환
            rotations_deg = np.degrees(rotations)
            frame_data.extend([rotations_deg[2], rotations_deg[1], rotations_deg[0]])  # ZYX 순서
        
        f.write(" ".join(f"{value:.6f}" for value in frame_data) + "\n")

def save_animation_to_bvh(filepath, animation_data, parent_indices, joint_offsets, 
                         joint_names=None, frame_time=0.0333333):
    """
    애니메이션 데이터를 BVH 파일로 저장
    
    Args:
        filepath: str - 저장할 BVH 파일 경로
        animation_data: np.ndarray - 애니메이션 데이터 [num_frames, num_joints, 3]
        parent_indices: list - 각 조인트의 부모 인덱스
        joint_offsets: np.ndarray - 각 조인트의 오프셋 [num_joints, 3]
        joint_names: list - 조인트 이름 리스트 (선택사항)
        frame_time: float - 프레임 간 시간 간격 (초)
    """
    # 스켈레톤 계층 구조 생성
    root_joint = create_skeleton_hierarchy(parent_indices, joint_offsets, joint_names)
    
    with open(filepath, 'w') as f:
        # 계층 구조 작성
        write_joint_hierarchy(f, root_joint)
        
        # 모션 데이터 작성
        write_motion_data(f, animation_data, frame_time)

def example_usage():
    # 테스트 데이터
    num_frames = 2
    num_joints = 3
    
    # 애니메이션 데이터 (2프레임, 3개 조인트, XYZ 회전)
    animation_data = np.array([
        # 프레임 1
        [[0, 0, np.pi/4],      # 루트 조인트
         [0, np.pi/6, 0],      # 조인트 1
         [np.pi/6, 0, 0]],     # 조인트 2
        # 프레임 2
        [[0, 0, np.pi/2],      # 루트 조인트
         [0, np.pi/3, 0],      # 조인트 1
         [np.pi/4, 0, 0]]      # 조인트 2
    ])
    
    # 부모 조인트 인덱스
    parent_indices = [-1, 0, 1]
    
    # 조인트 오프셋
    joint_offsets = np.array([
        [0, 0, 0],     # 루트 조인트
        [1, 0, 0],     # 조인트 1
        [1, 0, 0]      # 조인트 2
    ])
    
    # 조인트 이름
    joint_names = ["Root", "Joint1", "Joint2"]
    
    # BVH 파일로 저장
    save_animation_to_bvh(
        "test_animation.bvh",
        animation_data,
        parent_indices,
        joint_offsets,
        joint_names
    )
    print("BVH file has been created successfully!")

if __name__ == "__main__":
    example_usage()