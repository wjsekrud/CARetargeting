
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


import sys
sys.path.append("./")
from utils.NewVertex import load_from_simple_txt 
from utils.LBS import convert_weights_to_sparse
from utils.bvhparser import BVHParser ###
from aPyOpenGL.agl.bvh import BVH

class PrepDataloader(Dataset):
    def __init__(self, src_name, tgt_name):

        self.motion_sequences = []
        # 관절 이름과 인덱스 매핑
        #self.joint_name_to_idx = {}

        #self.joint_parents_list = []
        #self.joint_offsets_list = []

        src_v_path = Path(f"./dataset/vertexes/{src_name}_clean_vertices.txt")
        tgt_v_path = Path(f"./dataset/vertexes/{tgt_name}_clean_vertices.txt")
        self.src_vertices, self.src_tris, self.src_h = load_from_simple_txt(src_v_path)
        self.tgt_vertices, self.tgt_tris, self.tgt_h = load_from_simple_txt(tgt_v_path)
        #print(self.tgt_tris)

        #==========================================================\
        self.tgt_vert_pos = np.zeros((len(self.src_vertices), 3))
        self.tgt_vert_geo = np.zeros((len(self.tgt_vertices), 22))
        for i in range(len(self.tgt_vert_pos)):
            self.tgt_vert_pos[i] = self.src_vertices[i].position
            for weight in self.src_vertices[i].bone_weights:
                self.tgt_vert_geo[i][weight['bone_index']] = weight['weight'] # numpy 배열로 geometry 저장 [V, J]
        #==========================================================/

        src_bvh_paths = Path(f"./dataset/Animations/bvhs/{src_name}")
        src_bvh_files = list(src_bvh_paths.glob("*.bvh"))
        tgt_bvh_file = list(Path(f"./dataset/Animations/bvhs/{tgt_name}").glob("*.bvh"))[0] #character file
        print("target character geo file: ", tgt_bvh_file)
        
        #==========================================================\
        self.tgtchar_skel = BVH(str(tgt_bvh_file)).poses[0].skeleton
        tgtchar = self.tgtchar_skel.joints
        self.tgt_joint_offset = np.zeros((len(tgtchar),3), dtype=np.float32)
        self.tgt_joint_localrot = np.zeros((len(tgtchar),4), dtype=np.float32)
        self.joint_parents_list = self.tgtchar_skel.parent_idx

        for i in range(len(tgtchar)):
            self.tgt_joint_offset[i] = tgtchar[i].local_pos

        # BVH 파일들을 순회하며 데이터 로드
        for bvh_file in src_bvh_files:
            
            newpose = BVH(str(bvh_file)).poses
            self.motion_sequences.append(newpose)
            #self.motion_sequences.append(self._process_bvh_file(bvh_file))   ###
         #==========================================================/
        #self._initialize_joint_structure(tgt_bvh_file)


    '''
    def parsebvh(self, file):
        self.parser.parse_bvh(file)
        joint_rotations = self.parser.get_joint_rotations() # J : [F : 3]
        root_velocities = self.parser.get_root_velocity() # [F : 3]
        return joint_rotations, root_velocities
    #''

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
    #'''

    def __len__(self) -> int:
        return len(self.motion_sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray]:
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
        tris = []
        for tri in self.tgt_tris:
            tris.append(tri['indices'])

        return (
            self.motion_sequences[idx],
            self.tgt_vert_pos,
            self.tgt_vert_geo,
            None,
            np.array(self.tgt_joint_offset),
            self.tgtchar_skel,
            np.array(tris)
        )

def create_dataloader(
    src_name,
    tgt_name,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[PrepDataloader, list]:
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
    return dataset, dataset.joint_parents_list