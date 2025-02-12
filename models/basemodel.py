import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("./")
from utils.bvhexporter import save
#from aPyOpenGL.transforms.numpy import quat
from aPyOpenGL.transforms.torch import quat
from aPyOpenGL.agl.motion import Skeleton
from aPyOpenGL.agl.motion import Motion
from utils.LBS import linear_blend_skinning_2
from utils.tensorBVH import detect_mesh_collisions_parallel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

class BaseModel(nn.Module):
    def __init__(self, 
                 input_size, 
                 target_vpos, #버텍스 위치
                 target_skinning, # 스키닝 리스트
                 geo_embedding, # 스키닝 임베딩
                 target_offsets, # 본 오프셋 리스트
                 target_aPy_skel,
                 target_tris, # 폴리곤 인덱스
                 geo_embedding_size=64, # 스키닝 임베딩 사이즈
                 enc_hidden_size=512, 
                 dec_hidden_size=512, 
                 joint_positions_size=22*3, 
                 joint_config_size=22*3, 
                 root_velocity_size=3):
        super(BaseModel, self).__init__()
        
        self.h_enc = torch.zeros(2,1,enc_hidden_size).to(device)  # Encoder hidden state
        self.h_dec = torch.zeros(2,1,dec_hidden_size).to(device)  # Decoder hidden state
        self.prev_joint_positions = torch.zeros(66).to(device)
        self.prev_root_velocity = torch.zeros(3).to(device)
        self.global_positions = torch.zeros(66).to(device)
        self.skeleton = Skeleton()

        self.setupmesh(target_vpos, target_skinning, geo_embedding, target_offsets, target_aPy_skel, target_tris, None)

        self.clip, self.jclip = [], []
  
        self.encoder_gru = nn.GRU(
            input_size=input_size,  # 입력: [θᴬ, ωᴬ] (소스 캐릭터의 관절 각도와 각속도)
            hidden_size=enc_hidden_size,
            num_layers=2,
            batch_first=True,
            device=device
        )
        
        # 디코더 입력 크기: encoder 히든 스테이트 + 조인트 포지션 + 루트 속도 + 스키닝 임베딩, 조인트 configuration
        self.decoder_gru = nn.GRU(
            input_size=enc_hidden_size*2 + joint_positions_size + root_velocity_size + geo_embedding_size + joint_config_size,
            hidden_size=dec_hidden_size,
            num_layers=2,
            batch_first=True,
            device=device
        )

        self.rotation_layer = nn.Linear(dec_hidden_size, 88, device=device)
        self.velocity_layer = nn.Linear(dec_hidden_size, 3, device=device)


    def setupmesh(self, vpos, vskin, vskine, joint_offsets, Skeleton, tris, height):
        self.ref_vpos = vpos
        self.target_skinning = vskin,
        self.target_skinning = self.target_skinning[0]
        self.target_geo_embedding = vskine,
        self.target_geo_embedding = self.target_geo_embedding[0]
        self.target_joint_offset = joint_offsets,
        self.target_joint_offset = self.target_joint_offset[0]
        self.skeleton = Skeleton,
        self.skeleton = self.skeleton[0]
        self.ref_tris = tris
        self.height = height

    def encode(self, rotation, velocity):
        """
        source_motion: [Time, Joint idx, 3] + [Time, Joint len + 1, 3]
        """
            #print(rotation.shape, velocity.shape)
        source_motion = torch.tensor(np.concatenate((rotation.flatten(), velocity)),dtype=torch.float32).unsqueeze(1).to(device)
            #print("encoderinputshape",source_motion.transpose(0,1).shape)
        print(source_motion.device, self.h_enc.device)
        encoder_outputs, encoder_hidden = self.encoder_gru(source_motion.transpose(0,1).unsqueeze(0), self.h_enc)
        self.h_enc = encoder_hidden

        return encoder_outputs, encoder_hidden
    
    def decode(self):
        """
        encoder_hidden: 인코더의 마지막 히든 스테이트
        target_skeleton_embedding: PointNet으로 생성된 타깃 스켈레톤 임베딩
        """

        decoder_input = torch.cat([
            self.prev_joint_positions.flatten(),
            self.prev_root_velocity,
            self.target_joint_offset.flatten(),
            self.target_geo_embedding,
            self.h_enc.flatten(),
        ], dim=0).unsqueeze(0).unsqueeze(0)  # (batch_size, 1, decoder_input_size)
        
        # 디코더 실행
        output, decoder_hidden = self.decoder_gru(decoder_input, self.h_dec)
        self.h_dec = decoder_hidden

        local_rotation = self.rotation_layer(decoder_hidden)[-1].squeeze()
        root_velocity = self.velocity_layer(decoder_hidden)[-1].squeeze()
        
        return local_rotation, root_velocity
    
    def forward(self, sequence, prev_root):
        frame_rotations = sequence.local_quats
        frame_velocity = sequence.root_pos - prev_root

        self.encode(frame_rotations, frame_velocity)

        local_rotations, root_velocity = self.decode()
        joint_quat, joint_pos = quat.fk(local_rotations.reshape(-1,4), torch.tensor(sequence.root_pos), self.skeleton) #fk layer

        self.prev_root_velocity = root_velocity

        self.prev_joint_positions = joint_pos
        pos_reshaped = joint_pos.reshape(-1,3)

        rpos = np.array([sequence.root_pos[0], sequence.root_pos[1] * -0.5, sequence.root_pos[2]])
        global_positions = (pos_reshaped * root_velocity).flatten()
        vertex_positions = linear_blend_skinning_2(self.skeleton, 
                                                   self.skeleton.parent_idx, 
                                                   rpos, #y좌표만 줄여줘야 함
                                                   self.target_joint_offset, 
                                                   joint_quat,
                                                   self.ref_vpos, 
                                                   self.target_skinning).squeeze(0) #skinning layer
        vertex_positions = vertex_positions[0]
        vertex_positions[:,1][1] = vertex_positions[:,1][1] + sequence.root_pos[1] * 1.5

        self.global_positions = global_positions
        self.vertex_positions = vertex_positions

        return global_positions, vertex_positions
    
    def detect_frame_contacts(self, vpos):
        print(self.ref_vpos.shape, vpos.shape, self.ref_tris.shape)
        return detect_mesh_collisions_parallel(self.ref_vpos, self.ref_tris)#, self.height)

    def compute_geometry_energy(self, gpos, vpos, ref_contacts, contacts, lamb=0.5, beta=0.9, gamma=0.2):
        j2j = self.j2j_energy(vpos, ref_contacts)
        interp = self.int_energy(vpos, contacts)
        foot = self.foot_energy(gpos)

        return lamb * j2j + beta * interp + gamma * foot

    def j2j_energy(self, vpos, ref_contacts):
        V_len_inv = 1/len(ref_contacts)
        sum = 0
        for v1, v2 in ref_contacts:
            dx = vpos[v1][0] - vpos[v2][0]
            dy = vpos[v1][1] - vpos[v2][1]
            dz = vpos[v1][2] - vpos[v2][2]
            L2 = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            sum += L2
        return sum * V_len_inv

    def int_energy(self, vpos, contacts):
        Fsum = 0
        jsum = 0
        ksum = 0
        for p1, p2 in contacts:
            for idx in self.ref_tris[p1]:
                vpos = self.vertex_positions[idx]
                w = self.compare_geodesic_distance(vpos,p2)
                jsum += w * self.calc_distance_field(vpos, self.ref_tris)

            for idx in self.ref_tris[p2]:
                vpos = self.vertex_positions[idx]
                w = self.compare_geodesic_distance(vpos,p1)
                ksum += w * self.calc_distance_field(vpos, self.ref_tris)
            
            Fsum += jsum + ksum
        return Fsum

    def calc_distance_field(self, vertex, polygon):
        pass

    def compare_geodesic_distance(self, v1, v2):
        pass

    def foot_energy(self, gpos):
        pass

    def compute_skeleton_energy(self, omega=0.2):
        weak = self.weak_energy()
        ee = self.ee_energy()

        return weak + omega * ee

    def weak_energy(self):
        pass

    def ee_energy(self):
        pass



    def testfoward(self, sequence, glob_root):
        #print(sequence)
        frame_rotations = sequence.local_quats  # [num_joints, 4]
        frame_velocity = sequence.root_pos - sequence.root_pos # [3]
        #joints = []
        #for i in range(22):
        #    joint_quat, joint_pos = quat.fk(frame_rotations[i], self.target_joint_offset[0], self.skeleton[0])
        #    joints.append(quat.quaternion_to_euler(joint_quat))
        #self.jclip.append(joints)
        print(sequence.root_pos, self.target_joint_offset[0])
        '''
        self.skinned_verts = linear_blend_skinning(torch.tensor(sequence.skeleton.pre_xforms,dtype=torch.float32).to(device), 
                                                   sequence.skeleton.parent_idx, 
                                                   torch.tensor(sequence.root_pos,dtype=torch.float32).to(device), 
                                                   torch.tensor(self.target_joint_offset,dtype=torch.float32).to(device), 
                                                   torch.tensor(sequence.local_quats,dtype=torch.float32).unsqueeze(0).to(device), 
                                                   torch.tensor(self.ref_vpos,dtype=torch.float32).to(device), 
                                                   torch.tensor(self.target_skinning,dtype=torch.float32).to(device))
        #'''

        #'''
        self.skinned_verts = linear_blend_skinning_2(self.skeleton, 
                                                   self.skeleton.parent_idx, 
                                                   sequence.root_pos * -0.5,
                                                   self.target_joint_offset,
                                                   torch.tensor(frame_rotations,dtype=torch.float64).unsqueeze(0).to(device), 
                                                   self.ref_vpos, 
                                                   self.target_skinning).squeeze(0)
        #self.skinned_verts[:,1] = self.skinned_verts[:,1] + sequence.root_pos[1] * 1.5
        #'''
        


        #vertex_positions = self.skinning_layer(self.target_skinning[0], self.target_joint_offset[0], self.ref_vpos)

        #self.clip.append(vertex_positions)
        
    def printtype(self,t):
        print(type(t))

    
    def saveanim(self,pose):
        motion = Motion(pose)
        motion.export_as_bvh("C:/Users/vml/Documents/GitHub/CARetargeting/models/asdf.bvh")
        save_to_obj(self.skinned_verts.cpu().numpy(), self.ref_tris)

    
    def enc_optim(self):
        pass

        
def save_to_obj(vertices, tris, filepath="C:/Users/vml/Documents/GitHub/CARetargeting/models/output.obj"):
    """
    스키닝된 버텍스와 폴리곤 정보를 OBJ 파일로 저장합니다.
    
    Args:
        vertices: numpy.ndarray - 버텍스 위치 배열 (N x 3)
        tris: List[List[int]] - 삼각형을 구성하는 버텍스 인덱스 리스트 (CW 순서)
        filepath: str - 저장할 OBJ 파일 경로
    """
    print(vertices.shape)
    with open(filepath, 'w') as f:
        # 파일 헤더 작성
        f.write("# Exported by LBS Skinning Validator\n")
        f.write("# Vertices: {}\n".format(len(vertices)))
        f.write("# Faces: {}\n\n".format(len(tris)))
        #self.clip.append(vertex_positions)
        
        # 버텍스 정보 작성
        for vertex in vertices:
            #print(vertex)
            #print(vertex)
            # OBJ 파일은 Y-up 좌표계를 사용하므로, 필요한 경우 여기서 좌표계 변환을 수행할 수 있습니다
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(vertex[0], vertex[1], vertex[2]))
        
        f.write("\n")

         # 폴리곤 정보 작성
        # OBJ 파일의 인덱스는 1부터 시작하므로 1을 더해줍니다
        for tri in tris:
            #print(tri)
            # CW -> CCW 변환 (필요한 경우)
            # f.write("f {} {} {}\n".format(tri[0]+1, tri[2]+1, tri[1]+1))
            # CW 유지
            f.write("f {} {} {}\n".format(tri[0]+1, tri[1]+1, tri[2]+1))

    