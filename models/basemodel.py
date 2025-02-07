import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("./")
from utils.bvhexporter import save
from aPyOpenGL.transforms.numpy import quat
from aPyOpenGL.agl.motion import Skeleton
from aPyOpenGL.agl.motion import Motion
from utils.LBS import linear_blend_skinning, linear_blend_skinning_2
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
        
        self.h_enc = torch.zeros(2,1,enc_hidden_size)  # Encoder hidden state
        self.h_dec = torch.zeros(2,1,dec_hidden_size)  # Decoder hidden state
        self.prev_joint_positions = torch.zeros(66)
        self.prev_root_velocity = torch.zeros(3)
        self.global_positions = torch.zeros(66)
        self.skeleton = Skeleton()

        self.setupmesh(target_vpos, target_skinning, geo_embedding, target_offsets, target_aPy_skel, target_tris)

        self.clip, self.jclip = [], []
  
        self.encoder_gru = nn.GRU(
            input_size=input_size,  # 입력: [θᴬ, ωᴬ] (소스 캐릭터의 관절 각도와 각속도)
            hidden_size=enc_hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # 디코더 입력 크기: encoder 히든 스테이트 + 조인트 포지션 + 루트 속도 + 스키닝 임베딩, 조인트 configuration
        self.decoder_gru = nn.GRU(
            input_size=enc_hidden_size*2 + joint_positions_size + root_velocity_size + geo_embedding_size + joint_config_size,
            hidden_size=dec_hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.rotation_layer = nn.Linear(dec_hidden_size, 66)
        self.velocity_layer = nn.Linear(dec_hidden_size, 3)

    def setupmesh(self, vpos, vskin, vskine, joint_offsets, Skeleton, tris):
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

    def encode(self, rotation, velocity):
        """
        source_motion: [Time, Joint idx, 3] + [Time, Joint len + 1, 3]
        """
        #print(rotation.shape, velocity.shape)
        source_motion = torch.tensor(np.concatenate((rotation.flatten(), velocity)),dtype=torch.float32).unsqueeze(1)
        #print("encoderinputshape",source_motion.transpose(0,1).shape)

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
            torch.tensor(self.target_joint_offset.flatten(), dtype=torch.float32),
            self.target_geo_embedding,
            self.h_enc.flatten(),
        ], dim=0).unsqueeze(0).unsqueeze(0)  # (batch_size, 1, decoder_input_size)
        
        # 디코더 실행
        output, decoder_hidden = self.decoder_gru(decoder_input, self.h_dec)
        self.h_dec = decoder_hidden

        local_rotation = self.rotation_layer(decoder_hidden)[-1].squeeze()
        root_velocity = self.velocity_layer(decoder_hidden)[-1].squeeze()
        
        return local_rotation, root_velocity
    
    def forward(self, rotation, velocity):

        self.encode(rotation, velocity)

        local_rotations, root_velocity = self.decode()

        self.prev_root_velocity = root_velocity
        joint_positions = local_rotations

        self.prev_joint_positions = joint_positions

        pos_reshaped = joint_positions.reshape(-1,3)
        global_positions = (pos_reshaped * root_velocity).flatten()
        vertex_positions = self.skinning_layer(self.target_geo, joint_positions, self.ref_vertices)
        self.global_positions = global_positions
        self.vertex_positions = vertex_positions

        return global_positions, vertex_positions
    
    def testfoward(self, sequence, glob_root):
        #print(sequence)
        frame_rotations = sequence.local_quats  # [num_joints, 4]
        frame_velocity = sequence.root_pos - sequence.root_pos # [3]
        #joints = []
        #for i in range(22):
        #    joint_quat, joint_pos = quat.fk(frame_rotations[i], self.target_joint_offset[0], self.skeleton[0])
        #    joints.append(quat.quaternion_to_euler(joint_quat))
        #self.jclip.append(joints)
        
        self.skinned_verts = linear_blend_skinning(torch.tensor(sequence.skeleton.pre_xforms,dtype=torch.float32).to(device), 
                                                   sequence.skeleton.parent_idx, 
                                                   torch.tensor(sequence.root_pos,dtype=torch.float32).to(device), 
                                                   torch.tensor(self.target_joint_offset,dtype=torch.float32).to(device), 
                                                   torch.tensor(sequence.local_quats,dtype=torch.float32).unsqueeze(0).to(device), 
                                                   torch.tensor(self.ref_vpos,dtype=torch.float32).to(device), 
                                                   torch.tensor(self.target_skinning,dtype=torch.float32).to(device))
        #'''

        '''
        self.skinned_verts = linear_blend_skinning_2(self.skeleton, 
                                                   self.skeleton.parent_idx, 
                                                   sequence.root_pos,
                                                   torch.tensor(self.target_joint_offset,dtype=torch.float32).to(device), 
                                                   torch.tensor(frame_rotations,dtype=torch.float32).to(device), 
                                                   torch.tensor(self.ref_vpos,dtype=torch.float32).to(device), 
                                                   torch.tensor(self.target_skinning,dtype=torch.float32).to(device))
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
        for vertex in vertices[0]:
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

    