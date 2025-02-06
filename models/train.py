from tqdm import tqdm
import torch
from basemodel import BaseModel
from dataloader import PrepDataloader
from dataloader import create_dataloader

params = {
    'src_chars' : ['Remy'],
    'tgt_chars' : ['Remy']
}
def train_model(params, epoches=1000, warmup_epochs=50):
    
    input_size = 22 * 3 + 3

    FirstRun = True
    for i in range(min(len(params['src_chars']),len(params['tgt_chars']))): #캐릭터 조합에 따른 for 분기
        dataloader, parent_list = create_dataloader(params['src_chars'][i], params['tgt_chars'][i]) 

        num_motion = len(dataloader.motion_sequences)
        k = 0
        print(num_motion)
        print(f"training for src:{params['src_chars'][i]}, tgt:{params['tgt_chars'][i]}")
        #print(joint_mapping)
        for epoch in tqdm(range(epoches), desc="Epoch", leave=False): #epoch 설정에 따른 for 분기
            _, tgt_vert_pos, tgt_geo, geo_e, tgt_skel, tgt_tris = dataloader.__getitem__(0)

            if FirstRun:
                    model = BaseModel(input_size, tgt_vert_pos, tgt_skel, tgt_geo, 64, torch.zeros(64), tgt_tris, parent_list) #geo_e) 
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    FirstRun = False
            else:
                model.changeconfig(tgt_vert_pos, tgt_skel, tgt_geo, torch.zeros(64))#geo_e)

            batch_sequences, _, _, _, _, _ = dataloader.__getitem__(k) #각 애니메이션 클립으로부터 만들어진 batch로 인한 분기
            k = (k+1) % num_motion

            num_frames = len(batch_sequences)
            #print(num_frames)

            #print("start processing frames")
            for frame_idx in range(num_frames): #해당 애니메이션 내부의 각 프레임에서 처리되는 분기
                #print(f"frame idx: {frame_idx}")
                frame_rotations = batch_sequences[frame_idx]['rotations']  # [num_joints, 3]
                frame_velocity = batch_sequences[frame_idx]['root_velocity']  # [3]
                #print(batch_rotations, batch_velocities)
                
                # Process single frame
                #frame_gp, frame_vp = model(frame_rotations,frame_velocity)
                model.testfoward(frame_rotations,frame_velocity)
                #print(frame_rotations[0])
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
            
            model.saveanim()
            break
        break


            
def main():

    characters = [ "Y_bot", "Remy"]
    train_model(params)

    
if __name__ == "__main__":
    main()