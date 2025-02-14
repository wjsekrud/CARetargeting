
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from aPyOpenGL.agl.motion import Skeleton

import sys
sys.path.append("./")
from utils.NewVertex import load_from_simple_txt 
from utils.bvhparser import BVHParser ###
from aPyOpenGL.agl.bvh import BVH

class PrepDataloader(Dataset):
    def __init__(self, src_name, tgt_name):

        self.motion_sequences = []

        self.loadfiles(src_name,tgt_name)
        self.setvertinfo()

        self.tgtchar_skel = BVH(str(self.tgt_char_file),scale=1).poses[0].skeleton
        self.tgt_joint_offset = np.zeros((len(self.tgtchar_skel.joints),3), dtype=np.float32) # 조인트 오프셋
        #self.tgt_joint_localrot = np.zeros((len(self.tgtchar_skel.joints),4), dtype=np.float32) # 조인트 로테이션

        for i in range(len(self.tgtchar_skel.joints)):
            self.tgt_joint_offset[i] = self.tgtchar_skel.joints[i].local_pos
            print(self.tgt_joint_offset[i])

    def getmesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Skeleton, np.ndarray]:
        """
        Returns:
            (vpos, vskinnig, vskinning_e, joint_offsets, Skeleton, tris) 튜플
            - vpos [..., 3]
            - vskinning [..., J]
            - 
            - joint_offsets [J, 3]
            - Skeleton [Skeleton]
            - tris [..., 3]
        """
        tris = []
        for tri in self.tgt_tris:
            tris.append(tri['indices'])
        
        return (
            self.tgt_vert_pos,
            self.tgt_vert_skin,
            None,
            self.tgt_joint_offset,
            self.tgtchar_skel,
            np.array(tris),
            self.tgt_h,
            self.tgt_MGD
        )
    
    def getanim(self, idx:int):
        return (self.motion_sequences[idx], self.seq_contacts[idx])
    
    def getsrcinfo(self):
        return (self.src_h, )
    
    def loadfiles(self,src_name,tgt_name):
        
        tgt_v_path = Path(f"./dataset/vertexes/{tgt_name}_clean_vertices.txt")
        self.tgt_vertices, self.tgt_tris, self.tgt_h, self.tgt_MGD = load_from_simple_txt(tgt_v_path)
        src_anim_path = Path(f"./dataset/Animations/bvhs/{src_name}")
        src_contact_path = Path(f"./dataset/Contacts/{src_name}")

        src_v_path = Path(f"./dataset/vertexes/{src_name}_clean_vertices.txt")
        _, _, self.src_h, _ = load_from_simple_txt(src_v_path)

        self.src_anim_files = list(src_anim_path.glob("*.bvh"))
        self.tgt_char_file = list(Path(f"./dataset/Animations/bvhs/{tgt_name}").glob("*.bvh"))[0] #character file
        print("target character geo file: ", self.tgt_char_file)

        self.src_contact_files = list(src_contact_path.glob("*.txt"))

        for bvh_file in self.src_anim_files:           
            newpose = BVH(str(bvh_file)).poses
            self.motion_sequences.append(newpose) 

        self.seq_contacts = []
        for cont_file in self.src_contact_files:
            self.seq_contacts.append(self.parsecontactinfo(cont_file))
            
    def parsecontactinfo(self,file):
        frame_contacts = []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                frame_contacts.append([0])
                frame_contacts[-1][0] = int(line.split("|")[0])
                for pair in line.split("|")[1:-1]:
                    frame_contacts[-1].append((int(pair.split(",")[0]), int(pair.split(",")[1])))
        return frame_contacts


    def setvertinfo(self):
        self.tgt_vert_pos = np.zeros((len(self.tgt_vertices), 3))
        self.tgt_vert_skin = np.zeros((len(self.tgt_vertices), 22))
        for i in range(len(self.tgt_vert_pos)):
            self.tgt_vert_pos[i] = self.tgt_vertices[i].position
            for weight in self.tgt_vertices[i].bone_weights:
                self.tgt_vert_skin[i][weight['bone_index']] = weight['weight'] # numpy 배열로 geometry 저장 [V, J]

    