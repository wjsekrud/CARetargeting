import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import time
import argparse
from tqdm import tqdm

from PointNet import MeshAutoEncoder, FeatureLearningFramework
from PNdataloader import create_mesh_dataloader
from PNmodelsaver import ModelSaver
import random

def train(args):
    # 기본 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    
    # 데이터셋 로드
    train_loader, train_dataset = create_mesh_dataloader(
        mesh_dir=args.data_dir,
        max_vertices=args.max_vertices,
        batch_size=1,  # 배치 크기를 1로 고정
        num_joints=args.num_joints
    )
    
    # 사용할 캐릭터 목록 가져오기
    character_indices = list(range(len(train_dataset)))
    
    # 각 캐릭터의 데이터를 미리 GPU로 이동
    character_data = []
    for idx in character_indices:
        vertices, weights, mask, triangles, num_vertices = train_dataset[idx]
        vertices = vertices.to(device)
        weights = weights.to(device)
        mask = mask.to(device)
        
        # Joint positions 미리 계산
        joint_positions = train_dataset.get_joint_positions(idx)
        joint_positions = torch.FloatTensor(joint_positions).to(device)
        
        character_data.append({
            'vertices': vertices,
            'weights': weights,
            'mask': mask,
            'joint_positions': joint_positions,
            'num_vertices': num_vertices
        })
    
    # 모델 초기화
    framework = FeatureLearningFramework(
        num_joints=args.num_joints,
        feature_dim=args.feature_dim,
        learning_rate=args.learning_rate
    )
    framework.model.to(device)
    
    # 학습 루프
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        framework.model.train()
        epoch_losses = []
        
        # 이번 epoch에서 처리할 캐릭터 순서 랜덤화
        random.shuffle(character_indices)
        
        # 각 캐릭터에 대해 학습
        for char_idx in tqdm(character_indices, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            data = character_data[char_idx]
            
            # 학습 스텝 수행
            losses = framework.train_step(
                data['vertices'].unsqueeze(0),  # 배치 차원 추가
                data['weights'].unsqueeze(0),
                data['joint_positions'].unsqueeze(0)
            )
            
            current_loss = losses['total'].item()
            epoch_losses.append(current_loss)
            
            # Tensorboard 로깅
            writer.add_scalar('train/total_loss', current_loss, global_step)
            writer.add_scalar('train/reconstruction_loss', 
                            losses['reconstruction'].item(), global_step)

            
            global_step += 1
        
        # Epoch 종료 후 평균 loss 계산
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # 체크포인트 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            ModelSaver.save_checkpoint(
                framework=framework,
                epoch=epoch,
                loss=avg_loss,
                output_dir=output_dir,
                is_best=True,
                save_full=True
            )
    
    writer.close()
    print("Training completed!")

def main():
    parser = argparse.ArgumentParser(description='Train Mesh Auto-encoder')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default="./dataset/vertexes",
                        help='Directory containing mesh files')
    parser.add_argument('--max_vertices', type=int, default=21000,
                        help='Maximum number of vertices to support')
    parser.add_argument('--num_joints', type=int, default=22,
                        help='Number of joints in the skeleton')
    parser.add_argument('--file_pattern', type=str, default='*_clean_vertices.txt',
                        help='Pattern to match mesh files')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Dimension of feature vectors')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default="./dataset/PointNet",
                        help='Directory to save model checkpoints and logs')
    parser.add_argument('--save_full_model', action='store_true', default=True,
                        help='Save full model in addition to separate encoder/decoder')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()