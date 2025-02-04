import torch
import torch.nn as nn
import numpy as np
class BaseModel(nn.Module):
    def __init__(self, 
                 input_size, 
                 target_vertex, 
                 target_skel, 
                 target_geo,
                 geo_embedding_size, 
                 geo_embedding, 
                 enc_hidden_size=512, 
                 dec_hidden_size=512, 
                 joint_positions_size=22*3, 
                 joint_config_size=22*3, 
                 root_velocity_size=3):
        super(BaseModel, self).__init__()
        
        # State for maintaining temporal information
        self.h_enc = torch.zeros(enc_hidden_size).unsqueeze(1)  # Encoder hidden state
        self.h_dec = torch.zeros(dec_hidden_size)  # Decoder hidden state
        self.prev_joint_positions = None
        self.prev_root_velocity = None
        self.global_positions = None
        self.vertex_positions = None
        self.target_geo = target_geo
        self.target_skeleton = target_skel
        self.target_geo_embedding = geo_embedding
        self.target_vertices = target_vertex

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
            input_size=enc_hidden_size + joint_positions_size + root_velocity_size + geo_embedding_size + joint_config_size,
            hidden_size=dec_hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.rotation_layer = nn.Linear(dec_hidden_size, 3)
        self.velocity_layer = nn.Linear(dec_hidden_size, 3)

    def changeconfig(self, target_vertex, target_skel, target_geometry, geo_embedding):
        self.vertex_positions = target_vertex
        self.target_skeleton = target_skel
        self.target_geo = target_geometry
        self.target_geo_embedding = geo_embedding
        
        
    def encode(self, rotation, velocity):
        """
        source_motion: [Time, Joint idx, 3] + [Time, Joint len + 1, 3]
        """
        print(rotation.shape, velocity.shape)
        source_motion = torch.tensor(np.concatenate((rotation.flatten(), velocity)),dtype=torch.float32).unsqueeze(1)
        #print("tensorshape",source_motion)
        encoder_outputs, encoder_hidden = self.encoder_gru(source_motion, self.h_enc)
        self.h_enc = encoder_hidden
        return encoder_outputs, encoder_hidden
    
    def decode(self):
        """
        encoder_hidden: 인코더의 마지막 히든 스테이트
        target_skeleton_embedding: PointNet으로 생성된 타깃 스켈레톤 임베딩
        """

        # 초기 디코더 입력 준비
        decoder_input = torch.cat([
            self.prev_joint_positions,
            self.prev_root_velocity,
            self.target_skeleton,
            self.target_geo_embedding,
            self.h_enc,
        ], dim=1).unsqueeze(1)  # (batch_size, 1, decoder_input_size)
        
        # 디코더 실행
        output, decoder_hidden = self.decoder_gru(decoder_input, self.h_dec)
        self.h_dec = decoder_hidden

        local_rotations = self.rotation_layer(decoder_hidden)
        root_velocity = self.velocity_layer(decoder_hidden)
        
        return local_rotations, root_velocity
    
    def forward(self, rotation, velocity):
        """
        전체 모델의 순전파
        """
        # 인코딩
        print("Started encoding...")
        motion_features = self.encode(rotation, velocity)
        self.prev_local_rotation = motion_features[:-1]
        self.prev_root_velocity = motion_features[-1]
        print(f"done encoding. rot : {self.prev_local_rotation}, vel: {self.prev_root_velocity}")

        # 디코딩
        local_rotations, root_velocity = self.decode()

        self.prev_root_velocity = root_velocity

        joint_positions = self.fk_layer(local_rotations)
        self.prev_joint_positions = joint_positions

        global_positions = joint_positions + root_velocity
        vertex_positions = self.skinning_layer(self.target_geo, self.target_skeleton, self.target_vertices)
        self.global_positions = global_positions
        self.vertex_positions = vertex_positions

        return global_positions, vertex_positions
    
    def enc_optim():
        pass