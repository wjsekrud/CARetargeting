import numpy as np
from typing import Tuple, List, Dict
import re

class Joint:
    def __init__(self, name: str):
        self.name = name
        self.offset = np.zeros(3)
        self.channels = []
        self.children = []
        self.parent = None
        self.channel_start_idx = 0

def create_rotation_matrix(axis: str, angle: float) -> np.ndarray:
    """Create rotation matrix for given axis and angle (in degrees)"""
    angle = np.radians(angle)
    c = np.cos(angle)
    s = np.sin(angle)
    
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 'y':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    else:  # z
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

class BVHAnimation:
    def __init__(self, filename: str):
        self.joints: Dict[str, Joint] = {}
        self.joint_list: List[Joint] = []  # 순서가 있는 joint 리스트
        self.motion_data = None
        self.frame_time = 0.0
        self.load(filename)
    
    def load(self, filename: str):
        """BVH 파일 로드 및 파싱"""
        with open(filename, 'r') as f:
            content = f.read()
        
        hierarchy_part, motion_part = content.split('MOTION')
        self._parse_hierarchy(hierarchy_part)
        self._parse_motion(motion_part)
    
    def _parse_hierarchy(self, content: str):
        """HIERARCHY 섹션 파싱"""
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        stack = []
        channel_count = 0
        
        for line in lines:
            if 'HIERARCHY' in line:
                continue
                
            if 'ROOT' in line or 'JOINT' in line or 'End Site' in line:
                name = line.split()[-1]
                if 'End Site' in line:
                    name = f"end_site_{len(self.joints)}"
                    
                joint = Joint(name)
                self.joints[name] = joint
                self.joint_list.append(joint)
                
                if stack:  # Has parent
                    parent = stack[-1]
                    joint.parent = parent
                    parent.children.append(joint)
                    
                stack.append(joint)
                #print(joint)
                
            elif 'OFFSET' in line:
                offset = np.array([float(x) for x in line.split()[-3:]])
                stack[-1].offset = offset
                #print(offset)
                
            elif 'CHANNELS' in line:
                parts = line.split()
                num_channels = int(parts[1])
                channels = parts[2:]
                
                joint = stack[-1]
                joint.channels = channels
                joint.channel_start_idx = channel_count
                channel_count += num_channels
                
            elif '}' in line:
                stack.pop()
    
    def _parse_motion(self, content: str):
        """MOTION 섹션 파싱"""
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        
        num_frames = int(lines[0].split()[-1])
        self.frame_time = float(lines[1].split()[-1])
        
        motion_data = []
        for line in lines[2:]:
            if line:
                frame_data = [float(x) for x in line.split()]
                motion_data.append(frame_data)
                #print(frame_data)
        
        self.motion_data = np.array(motion_data)
    
    def get_joint_matrices(self, frame_idx: int) -> List[np.ndarray]:
        """특정 프레임에서의 모든 joint transformation matrices 반환"""
        frame_data = self.motion_data[frame_idx]
        matrices = []
        
        def process_joint(joint: Joint, parent_matrix: np.ndarray) -> None:
            # Local transformation matrix
            local_matrix = np.eye(4)
            
            # Set translation from offset
            local_matrix[:3, 3] = joint.offset
            
            # Apply channel transformations
            channel_values = frame_data[joint.channel_start_idx:
                                      joint.channel_start_idx + len(joint.channels)]
            
            rotation_matrix = np.eye(3)
            translation = np.zeros(3)
            
            for channel, value in zip(joint.channels, channel_values):
                channel = channel.lower()
                if 'position' in channel:
                    idx = 'xyz'.index(channel[0])
                    translation[idx] = value
                else:  # rotation
                    axis = channel[0]
                    rotation = create_rotation_matrix(axis, value)
                    rotation_matrix = rotation_matrix @ rotation
            
            # Apply transformations
            local_matrix[:3, :3] = rotation_matrix
            local_matrix[:3, 3] += translation
            
            # Calculate global matrix
            global_matrix = parent_matrix @ local_matrix
            matrices.append(global_matrix.astype(np.float32))
            
            # Process children
            for child in joint.children:
                process_joint(child, global_matrix)
        
        # Start from root
        root = self.joint_list[0]
        process_joint(root, np.eye(4))
        
        return matrices
    
    def get_joint_names(self) -> List[str]:
        """모든 joint 이름 반환 (인덱스 순서대로)"""
        return [joint.name for joint in self.joint_list]
    
    @property
    def num_frames(self) -> int:
        """총 프레임 수 반환"""
        return len(self.motion_data)