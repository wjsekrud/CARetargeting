from tqdm import tqdm
import torch
from basemodel import BaseModel
from dataloader import PrepDataloader

params = {
    'src_chars' : ['Remy'],
    'tgt_chars' : ['Remy']
}
def train_model(params, epoches=1000, warmup_epochs=50):
    
    input_size = 22 * 3 + 3
    model = BaseModel(input_size,None,None,None,None,None,None)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(min(len(params['src_chars']),len(params['tgt_chars']))): #캐릭터 조합에 따른 for 분기
        
        dataloader = PrepDataloader(params['src_chars'][i],params['tgt_chars'][i])
        num_motion = len(dataloader.motion_sequences)
        
        print(f"training for src:{params['src_chars'][i]}, tgt:{params['tgt_chars'][i]}")

        vpos, vskin, _, joint_offsets, Skeleton, tris = dataloader.getmesh()
        model.setupmesh(vpos,vskin,None,joint_offsets,Skeleton,tris)

        k = 0
        for epoch in tqdm(range(epoches), desc="Epoch", leave=False): #epoch 설정에 따른 for 분기

            batch_sequences = dataloader.getanim(k) #각 애니메이션 클립으로부터 만들어진 batch로 인한 분기
            k = (k+1) % num_motion

            num_frames = len(batch_sequences)

            for frame_idx in range(num_frames): #해당 애니메이션 내부의 각 프레임에서 처리되는 분기

                #======================================================================\
                frame_rotations = batch_sequences[frame_idx].local_quats  # [num_joints, 4]
                frame_velocity = batch_sequences[max(frame_idx-1,0)].root_pos - batch_sequences[frame_idx].root_pos # [3]
                #print(frame_velocity)
                #======================================================================/
                
                # Process single frame
                #frame_gp, frame_vp = model(batch_sequences[0], batch_sequences[0].root_pos)
                model.testfoward(batch_sequences[0], batch_sequences[0].root_pos)

                '''
                # Detect contacts for current frame
                frame_contacts = model.detect_frame_contacts(frame_vp)
                
                # Compute loss for current frame
                geom_energy = model.compute_geometry_energy(
                    frame_gp,
                    frame_vp, 
                    frame_contacts,
                    tgt_vert_pos
                )

                skel_energy = model.compute_skeleton_energy(
                    model.prev_joint_positions,
                    model.global_positions
                )

                frame_energy = geom_energy + skel_energy
                #'''
                #frame_energy = torch.nn.MSELoss()
                # Backward pass and optimize
                optimizer.zero_grad()
                #frame_energy.backward()
                optimizer.step()
                #print("Step")
                # Frame-wise encoder space optimization
                if epoch > warmup_epochs:
                    frame_output = model.enc_optim()
            
                model.saveanim(batch_sequences)
                break
            break
        break


            
def main():

    characters = [ "Y_bot", "Remy"]
    train_model(params)

    
if __name__ == "__main__":
    main()