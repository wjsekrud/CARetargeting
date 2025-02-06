import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("./")
from utils.LBS import convert_weights_to_sparse
from utils.FKlayer import compute_bone_positions
from utils.bvhexporter import save_animation_to_bvh

class BaseModel(nn.Module):
    def __init__(self, 
                 input_size, 
                 target_vertex, 
                 target_skel, 
                 target_geo,
                 geo_embedding_size, 
                 geo_embedding, 
                 target_tris,
                 parent_list,
                 enc_hidden_size=512, 
                 dec_hidden_size=512, 
                 joint_positions_size=22*3, 
                 joint_config_size=22*3, 
                 root_velocity_size=3):
        super(BaseModel, self).__init__()
        
        # State for maintaining temporal information
        self.h_enc = torch.zeros(2,1,enc_hidden_size)  # Encoder hidden state
        self.h_dec = torch.zeros(2,1,dec_hidden_size)  # Decoder hidden state
        self.prev_joint_positions = torch.zeros(66)
        self.prev_root_velocity = torch.zeros(3)
        self.global_positions = torch.zeros(66)
        self.vertex_positions = None
        self.target_geo = target_geo
        self.target_skeleton = target_skel #[22,3]
        self.target_geo_embedding = geo_embedding
        self.ref_vertices = target_vertex
        self.ref_tris = target_tris
        self.target_geo_sps = convert_weights_to_sparse(target_geo,len(self.ref_vertices),len(self.target_skeleton))

        self.clip = []
        self.jclip = []
        self.parent_list = parent_list
        # 인코더 GRU
        self.encoder_gru = nn.GRU(
            input_size=input_size,  # 입력: [θᴬ, ωᴬ] (소스 캐릭터의 관절 각도와 각속도)
            hidden_size=enc_hidden_size,
            num_layers=2,
            batch_first=True
        )
        print("encodersize", self.encoder_gru.input_size)
        
        # 디코더 GRU
        # 디코더 입력 크기: encoder 히든 스테이트 + 조인트 포지션 + 루트 속도 + 스키닝 임베딩, 조인트 configuration
        self.decoder_gru = nn.GRU(
            input_size=enc_hidden_size*2 + joint_positions_size + root_velocity_size + geo_embedding_size + joint_config_size,
            hidden_size=dec_hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.rotation_layer = nn.Linear(dec_hidden_size, 66)
        self.velocity_layer = nn.Linear(dec_hidden_size, 3)

    def changeconfig(self, target_vertex, target_skel, target_geometry, geo_embedding):
        self.ref_vertices = target_vertex
        self.target_skeleton = target_skel
        self.target_geo = target_geometry
        self.target_geo_embedding = geo_embedding
        self.target_geo_sps = convert_weights_to_sparse(self.target_geo,len(self.ref_vertices),len(self.target_skeleton))
        
        
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

        # 초기 디코더 입력 준비
        #motion = np.concatenate((self.prev_joint_positions.flatten(), self.prev_root_velocity))
        #print("decoderinputshade", self.prev_joint_positions.flatten().shape, self.prev_root_velocity.shape)
        decoder_input = torch.cat([
            self.prev_joint_positions.flatten(),
            self.prev_root_velocity,
            torch.tensor(self.target_skeleton.flatten(), dtype=torch.float32),
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
        """
        전체 모델의 순전파
        """
        # 인코딩
        #print("Started encoding...")
        self.encode(rotation, velocity)

        # 디코딩
        #print("Started Decoding...")
        local_rotations, root_velocity = self.decode()

        self.prev_root_velocity = root_velocity
        #print(self.prev_root_velocity[-1])

        #joint_positions = self.fk_layer(local_rotations[-1]) # [joint, 3]
        joint_positions = local_rotations

        self.prev_joint_positions = joint_positions

        pos_reshaped = joint_positions.reshape(-1,3)
        global_positions = (pos_reshaped * root_velocity).flatten()
        vertex_positions = self.skinning_layer(self.target_geo, joint_positions, self.ref_vertices)
        self.global_positions = global_positions
        self.vertex_positions = vertex_positions

        return global_positions, vertex_positions
    
    def testfoward(self, rotation, velocity):

        joint_positions = compute_bone_positions(rotation, self.target_skeleton, self.parent_list )
        vertex_positions = self.skinning_layer(self.target_geo, joint_positions, self.ref_vertices)

        self.clip.append(vertex_positions)
        self.jclip.append(rotation)
        #print(len(self.jclip),len(self.jclip[-1]))
        #print(self.jclip)
    
    
    def saveanim(self):
        save_to_obj(self.clip[0], self.ref_tris)
        save_animation_to_bvh("C:/Users/vml/Documents/GitHub/CARetargeting/models/output.bvh",
                              np.array(self.jclip), self.parent_list ,self.target_skeleton)
        print("done saveanim")
        
    
    def enc_optim(self):
        pass

    def skinning_layer(self, target_geometry, bone_transforms, vertices):

        # 결과를 저장할 배열 초기화
        num_vertices = len(vertices)
        skinned_vertices = np.zeros_like(vertices)
        
        # 각 버텍스에 대해 스키닝 계산
        for vertex_idx in range(num_vertices):
            vertex_pos = vertices[vertex_idx]
            weights_info = target_geometry[vertex_idx]
            
            # 현재 버텍스의 최종 위치를 계산
            blended_position = np.zeros(3)
            
            # 각 영향받는 본에 대해 가중치를 적용하여 위치 계산
            for bone_info in weights_info:
                bone_idx = bone_info['bone_index']
                weight = bone_info['weight']
                
                # 본 오프셋을 이용한 변환 계산
                bone_offset = bone_transforms[bone_idx]
                
                # 본 변환 행렬 계산 (여기서는 간단한 이동 변환만 적용)
                transformed_position = vertex_pos + bone_offset
                
                # 가중치를 적용하여 최종 위치에 더함
                blended_position += weight * transformed_position
                
            skinned_vertices[vertex_idx] = blended_position
            
        return skinned_vertices
    
    


def save_to_obj(vertices, tris, filepath="C:/Users/vml/Documents/GitHub/CARetargeting/models/output.obj"):
    """
    스키닝된 버텍스와 폴리곤 정보를 OBJ 파일로 저장합니다.
    
    Args:
        vertices: numpy.ndarray - 버텍스 위치 배열 (N x 3)
        tris: List[List[int]] - 삼각형을 구성하는 버텍스 인덱스 리스트 (CW 순서)
        filepath: str - 저장할 OBJ 파일 경로
    """
    with open(filepath, 'w') as f:
        # 파일 헤더 작성
        f.write("# Exported by LBS Skinning Validator\n")
        f.write("# Vertices: {}\n".format(len(vertices)))
        f.write("# Faces: {}\n\n".format(len(tris)))
        
        # 버텍스 정보 작성
        for vertex in vertices:
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
