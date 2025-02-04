
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


import sys
sys.path.append("./")
from utils.NewVertex import load_from_simple_txt 
from utils.bvhparser import BVHParser


class PrepDataloader(Dataset):
    def __init__(self, src_name, tgt_name):

        self.motion_sequences = []
        # 관절 이름과 인덱스 매핑 생성
        self.joint_name_to_idx = {}
        self.joint_parents_list = []
        self.joint_offsets_list = []

        src_v_path = Path(f"./dataset/vertexes/{src_name}_clean_vertices.txt")
        tgt_v_path = Path(f"./dataset/vertexes/{tgt_name}_clean_vertices.txt")
        self.src_vertices, self.src_tris, self.src_h = load_from_simple_txt(src_v_path)
        self.tgt_vertices, self.tgt_tris, self.tgt_h = load_from_simple_txt(tgt_v_path)

        self.tgt_vert_pos = []
        self.tgt_vert_geo = []
        for verts in self.tgt_vertices:
            self.tgt_vert_pos.append(verts.position)
            self.tgt_vert_geo.append(verts.bone_weights)

        src_bvh_paths = Path(f"./dataset/Animations/bvhs/{src_name}")
        src_bvh_files = list(src_bvh_paths.glob("*.bvh"))
        tgt_bvh_file = list(Path(f"./dataset/Animations/bvhs/{tgt_name}").glob("*.bvh"))[0]
        print("tgt_bvh_file: ", tgt_bvh_file)
        
         # BVH 파일들을 순회하며 데이터 로드
        first_file = True
        for bvh_file in src_bvh_files:
            if first_file:
                # 첫 번째 파일에서 관절 구조 정보 초기화
                self._initialize_joint_structure(bvh_file)
                first_file = False

            self.motion_sequences.append(self._process_bvh_file(bvh_file))   

        self._initialize_joint_structure(tgt_bvh_file)
    
    def _initialize_joint_structure(self, bvh_path: str):
        """
        첫 번째 BVH 파일을 기반으로 관절 구조 초기화
        """
        
        print("bvh_path = ", bvh_path)
        parser = BVHParser()
        parser.parse_bvh(bvh_path)
        
        # 관절 이름과 인덱스 매핑 생성
        for idx, joint_name in enumerate(parser.joint_channels.keys()):
            self.joint_name_to_idx[joint_name] = idx
            #print(idx, joint_name, self.joint_name_to_idx[joint_name])
        
        # 부모 관절 정보를 리스트로 변환
        num_joints = len(self.joint_name_to_idx)
        self.joint_parents_list = [-1] * num_joints  # -1은 부모가 없음을 의미
        
        for joint_name, parent_name in parser.joint_parents.items():
            if joint_name[-3:] not in ["JOINT", "End", "ROOT"]:
                joint_idx = self.joint_name_to_idx[joint_name]
                parent_idx = self.joint_name_to_idx[parent_name]
            self.joint_parents_list[joint_idx] = parent_idx
        
        # 오프셋 정보를 리스트로 변환
        self.joint_offsets_list = [None] * num_joints
        for joint_name, offset in parser.joint_offsets.items():
            if joint_name[-3:] not in ["JOINT", "End", "ROOT"]:
                joint_idx = self.joint_name_to_idx[joint_name]
                self.joint_offsets_list[joint_idx] = offset

    '''
    def parsebvh(self, file):
        self.parser.parse_bvh(file)
        joint_rotations = self.parser.get_joint_rotations() # J : [F : 3]
        root_velocities = self.parser.get_root_velocity() # [F : 3]
        return joint_rotations, root_velocities
    #'''

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
        sequences = []
        for frame_idx in range(num_frames):
            sequence_data = {
                'rotations': frame_rotations[frame_idx],  # Shape: [num_joints, 3]
                'root_velocity': root_velocity[frame_idx]  # Shape: [3]
            }
            #print(sequence_data)
            sequences.append(sequence_data)

        return sequences
    
    def __len__(self) -> int:
        return len(self.motion_sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
        """
        데이터셋의 단일 항목 반환
        
        Returns:
            (joint_rotations, root_velocity, tgt_vertices, tgt_geometries, tgt_geo_e, tgt_skeleon) 튜플
            - joint_rotations: Shape [num_joints, 3]
            - root_velocity: Shape [3]
            - tgt_vertices: [num_vertices, 3]
            - tgt_geometries: [num_vertices, num_skin_weights] (mesh skinning weights)
            - tgt_geo_e
            - tgt_skeleon: [num_joints, 3] (bone offsets)
        """

        #TODO: 버텍스 정보 반환하도록 변경, PointNet 임베딩 같이 반환하도록 구현
        return (
            self.motion_sequences[idx],
            np.array(self.tgt_vert_pos),
            self.tgt_vert_geo,
            None,
            np.array(self.joint_offsets_list)
        )

def create_dataloader(
    src_name,
    tgt_name,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[PrepDataloader, Dict[str, int]]:
    """
    모션 데이터 로더 생성
    
    Args:
        bvh_folder: BVH 파일들이 있는 폴더 경로
        batch_size: 배치 크기
        shuffle: 데이터 셔플 여부
        
    Returns:
        DataLoader 객체
    """
    dataset = PrepDataloader(src_name,tgt_name)
    return dataset, dataset.joint_name_to_idx