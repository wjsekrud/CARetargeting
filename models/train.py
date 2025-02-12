from tqdm import tqdm
import torch
from basemodel import BaseModel
from dataloader import PrepDataloader

params = {
    'src_chars' : ['Remy'],
    'tgt_chars' : ['Remy']
}
def train_model(params, epoches=1000, warmup_epochs=50):
    
    input_size = 22 * 4 + 3
    model = BaseModel(input_size,None,None,None,None,None,None)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(min(len(params['src_chars']),len(params['tgt_chars']))): #캐릭터 조합에 따른 for 분기
        
        dataloader = PrepDataloader(params['src_chars'][i],params['tgt_chars'][i])
        num_motion = len(dataloader.motion_sequences)
        
        print(f"training for src:{params['src_chars'][i]}, tgt:{params['tgt_chars'][i]}")

        vpos, vskin, _, joint_offsets, Skeleton, tris, height = dataloader.getmesh()
        model.setupmesh(torch.tensor(vpos).to(device="cuda"),
                        torch.tensor(vskin).to(device="cuda"),
                        torch.zeros(64).to(device="cuda"),
                        torch.tensor(joint_offsets).to(device="cuda"),
                        Skeleton,
                        torch.tensor(tris).to(device="cuda"),
                        height)

        k = 0
        for epoch in tqdm(range(epoches), desc="Epoch", leave=False): #epoch 설정에 따른 for 분기

            batch_sequences, batch_contacts = dataloader.getanim(k) #각 애니메이션 클립으로부터 만들어진 batch로 인한 분기
            k = (k+1) % num_motion

            num_frames = len(batch_sequences)

            for frame_idx in range(num_frames): #해당 애니메이션 내부의 각 프레임에서 처리되는 분기
                
                # Process single frame
                #frame_gp, frame_vp = model(batch_sequences[frame_idx], batch_sequences[max(frame_idx-1,0)].root_pos)
                #
                model.testfoward(batch_sequences[0], batch_sequences[0].root_pos)

                
                # Detect contacts for current frame

                '''
                print(f"exec dfc at frame{frame_idx}")
                frame_contacts = model.detect_frame_contacts(frame_vp)

                if frame_contacts != []:
                    print(f"contacts: {len(frame_contacts)}")

                for contacts in batch_contacts:
                    if frame_idx == contacts[0]:
                        ref_contacts = contacts[1:]
                        break
                
                
                # Compute loss for current frame
                
                geom_energy = model.compute_geometry_energy(
                    frame_gp,
                    frame_vp, 
                    ref_contacts,
                    frame_contacts
                )
                
                skel_energy = model.compute_skeleton_energy(
                    model.prev_joint_positions,
                    model.global_positions
                )

                frame_energy = geom_energy + skel_energy
                #'''

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