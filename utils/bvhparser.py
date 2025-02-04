import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import sys
sys.path.append("./")
class BVHParser:
    def __init__(self):
        self.joint_channels: Dict[str, List[str]] = {}  # 각 관절의 채널 정보
        self.joint_parents: Dict[str, str] = {}         # 각 관절의 부모 관절
        self.joint_offsets: Dict[str, np.ndarray] = {}  # 각 관절의 오프셋
        self.frames: int = 0                            # 총 프레임 수
        self.frame_time: float = 0                      # 프레임 간격
        self.motion_data: np.ndarray = None            # 모션 데이터

    def parse_bvh(self, file_path: str) -> None:
        """BVH 파일을 파싱합니다."""
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # HIERARCHY와 MOTION 섹션 분리
        hierarchy_end = lines.index('MOTION\n')
        hierarchy_lines = lines[:hierarchy_end]
        motion_lines = lines[hierarchy_end+1:]
        
        self._parse_hierarchy(hierarchy_lines)
        self._parse_motion(motion_lines)

    def _parse_hierarchy(self, lines: List[str]) -> None:
        """HIERARCHY 섹션을 파싱합니다."""
        current_joint = None
        joint_stack = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            tokens = line.split()
            
            if tokens[0] in ["JOINT", "End", "ROOT"]:
                joint_name = tokens[1] if tokens[0] != "End" else f"{current_joint}_End"
                current_joint = joint_name
                
                if joint_stack:
                    self.joint_parents[current_joint] = joint_stack[-1]
                joint_stack.append(current_joint)
                
            elif tokens[0] == "OFFSET":
                self.joint_offsets[current_joint] = np.array([float(x) for x in tokens[1:4]])
                
            elif tokens[0] == "CHANNELS":
                num_channels = int(tokens[1])
                self.joint_channels[current_joint] = tokens[2:2+num_channels]
                
            elif tokens[0] == "}":
                joint_stack.pop()
                if joint_stack:
                    current_joint = joint_stack[-1]

    def _parse_motion(self, lines: List[str]) -> None:
        """MOTION 섹션을 파싱합니다."""
        # 프레임 수와 프레임 시간 파싱
        self.frames = int(lines[0].split()[-1])
        self.frame_time = float(lines[1].split()[-1])
        
        # 모션 데이터 파싱
        motion_data = []
        for line in lines[2:]:
            if line.strip():
                values = [float(x) for x in line.strip().split()]
                motion_data.append(values)
        
        self.motion_data = np.array(motion_data)

    def get_joint_rotations(self) -> Dict[str, np.ndarray]:
        """각 프레임별 관절 회전값을 반환합니다."""
        joint_rotations = {}
        channel_index = 0
        
        for joint_name, channels in self.joint_channels.items():
            num_channels = len(channels)
            if "rotation" in " ".join(channels).lower():
                # XYZ 회전 채널만 추출
                rot_indices = [
                    i + channel_index 
                    for i, ch in enumerate(channels) 
                    if "rotation" in ch.lower()
                ]
                
                if rot_indices:
                    rotations = self.motion_data[:, rot_indices]
                    joint_rotations[joint_name] = rotations
            
            channel_index += num_channels
            
        return joint_rotations

    def get_root_velocity(self) -> np.ndarray:
        """각 프레임별 루트 조인트의 속도를 계산합니다."""
        # 루트 조인트의 위치 채널 찾기
        root_joint = list(self.joint_channels.keys())[0]  # 첫 번째 조인트가 루트
        channels = self.joint_channels[root_joint]
        
        pos_indices = [
            i for i, ch in enumerate(channels) 
            if ch.lower().endswith('position')
        ]
        
        if not pos_indices:
            raise ValueError("루트 조인트에 위치 정보가 없습니다.")
            
        # 위치 데이터 추출
        positions = self.motion_data[:, pos_indices]
        
        # 속도 계산 (프레임 간 차이)
        velocities = np.zeros_like(positions)
        velocities[1:] = (positions[1:] - positions[:-1]) / self.frame_time
        
        return velocities

def main():
    # 사용 예시
    parser = BVHParser()
    parser.parse_bvh(Path("./animation.bvh"))
    
    # 관절 회전값 추출
    joint_rotations = parser.get_joint_rotations()
    print("Joint rotations shape for each joint:")
    #for joint, rotations in joint_rotations.items():
    #    print(f"{joint}: {rotations.shape}")
    
    # 루트 속도 추출
    root_velocity = parser.get_root_velocity()
    print(f"\nRoot velocity shape: {root_velocity.shape}")

if __name__ == "__main__":
    main()