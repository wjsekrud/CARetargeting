import numpy as np
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import os

class BVHMotionDataset(Dataset):
    def __init__(self, bvh_folder: str):
        """
        여러 BVH 파일들을 처리하여 최적화된 리스트 기반 데이터셋으로 변환하는 클래스
        
        Args:
            bvh_folder: BVH 파일들이 있는 폴더 경로
        """
        self.motion_sequences = []
        
        # 관절 이름과 인덱스 매핑 생성
        self.joint_name_to_idx = {}
        self.joint_parents_list = []
        self.joint_offsets_list = []
        
        # BVH 파일들을 순회하며 데이터 로드
        first_file = True
        for bvh_file in os.listdir(bvh_folder):
            if bvh_file.endswith('.bvh'):
                file_path = os.path.join(bvh_folder, bvh_file)
                
                if first_file:
                    # 첫 번째 파일에서 관절 구조 정보 초기화
                    self._initialize_joint_structure(file_path)
                    first_file = False
                
                self._process_bvh_file(file_path)

    def _initialize_joint_structure(self, bvh_path: str):
        """
        첫 번째 BVH 파일을 기반으로 관절 구조 초기화
        """
        parser = BVHParser()
        parser.parse_bvh(bvh_path)
        
        # 관절 이름과 인덱스 매핑 생성
        for idx, joint_name in enumerate(parser.joint_channels.keys()):
            self.joint_name_to_idx[joint_name] = idx
        
        # 부모 관절 정보를 리스트로 변환
        num_joints = len(self.joint_name_to_idx)
        self.joint_parents_list = [-1] * num_joints  # -1은 부모가 없음을 의미
        
        for joint_name, parent_name in parser.joint_parents.items():
            joint_idx = self.joint_name_to_idx[joint_name]
            parent_idx = self.joint_name_to_idx[parent_name]
            self.joint_parents_list[joint_idx] = parent_idx
        
        # 오프셋 정보를 리스트로 변환
        self.joint_offsets_list = [None] * num_joints
        for joint_name, offset in parser.joint_offsets.items():
            joint_idx = self.joint_name_to_idx[joint_name]
            self.joint_offsets_list[joint_idx] = offset

    def _process_bvh_file(self, bvh_path: str):
        """
        단일 BVH 파일을 처리하여 최적화된 프레임 시퀀스로 변환
        """
        parser = BVHParser()
        parser.parse_bvh(bvh_path)
        
        # 관절 회전값과 루트 속도 추출
        joint_rotations = parser.get_joint_rotations()
        root_velocity = parser.get_root_velocity()
        
        num_frames = len(root_velocity)
        num_joints = len(self.joint_name_to_idx)
        
        # 모든 프레임의 회전값을 하나의 배열로 통합
        # Shape: [num_frames, num_joints, 3] (XYZ rotation per joint)
        frame_rotations = np.zeros((num_frames, num_joints, 3))
        
        for joint_name, rotations in joint_rotations.items():
            joint_idx = self.joint_name_to_idx[joint_name]
            frame_rotations[:, joint_idx] = rotations
        
        # 각 프레임을 처리하여 시퀀스 데이터 생성
        for frame_idx in range(num_frames):
            sequence_data = {
                'rotations': frame_rotations[frame_idx],  # Shape: [num_joints, 3]
                'root_velocity': root_velocity[frame_idx]  # Shape: [3]
            }
            self.motion_sequences.append(sequence_data)

    def __len__(self) -> int:
        return len(self.motion_sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        데이터셋의 단일 항목 반환
        
        Returns:
            (joint_rotations, root_velocity, joint_parents, joint_offsets) 튜플
            - joint_rotations: Shape [num_joints, 3]
            - root_velocity: Shape [3]
            - joint_parents: Shape [num_joints]
            - joint_offsets: Shape [num_joints, 3]
        """
        sequence = self.motion_sequences[idx]
        return (
            sequence['rotations'],
            sequence['root_velocity'],
            np.array(self.joint_parents_list),
            np.array(self.joint_offsets_list)
        )

def create_motion_dataloader(
    bvh_folder: str,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[DataLoader, Dict[str, int]]:
    """
    최적화된 모션 데이터 로더 생성
    
    Returns:
        (DataLoader 객체, 관절 이름-인덱스 매핑 딕셔너리) 튜플
    """
    dataset = BVHMotionDataset(bvh_folder)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    ), dataset.joint_name_to_idx

def train_model(
    bvh_folder: str,
    params,
    epochs: int = 20000,
    warmup_epochs: int = 500,
    batch_size: int = 32
):
    """
    최적화된 모델 학습 함수
    """
    model = BaseModel()
    optimizer = torch.optim.Adam(params)
    dataloader, joint_mapping = create_motion_dataloader(bvh_folder, batch_size)
    
    for epoch in tqdm(range(epochs), desc="Epoch", leave=False):
        for batch_rotations, batch_velocities, batch_parents, batch_offsets in dataloader:
            model.reset_states()
            
            # 배치 내의 각 프레임 처리
            num_frames = batch_rotations.shape[1]
            for frame_idx in range(num_frames):
                # 현재 프레임의 데이터 추출
                frame_rotations = batch_rotations[:, frame_idx]  # [batch_size, num_joints, 3]
                frame_velocity = batch_velocities[:, frame_idx]  # [batch_size, 3]
                
                # 단일 프레임 처리
                frame_gp, frame_vp = model.process_single_frame(
                    frame_rotations,
                    frame_velocity,
                    batch_parents,
                    batch_offsets
                )