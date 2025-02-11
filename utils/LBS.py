import os
import torch
import numpy as np

from OpenGL.GL import * 
from aPyOpenGL.transforms import torch as trf_t

def linear_blend_skinning(pre_xforms, parents, root_pos, bone_offset, quats, rest_verts_pos, skin_weights):
    '''
        pre_xforms       (J, 4, 4)
        parents          (J, )
        root pos         (3,)
        bone_offset      (J, 3)
        quats            (T, J, 4)
        rest_verts_pos   (num_vertices, 3), initial vertex positions
        skin_weights     (num_vertices, J)
    '''
    # T, J, num_of_verts 정의 
    T = quats.shape[0]
    J = bone_offset.shape[0]  # J
    nov = rest_verts_pos.shape[0]
    
    # pre-xforms 적용
    pre_xforms_cl = pre_xforms.clone()      # (J, 4, 4)
    print(pre_xforms)
    pre_quats  = trf_t.xform.to_quat(pre_xforms_cl)                                    # (J, 4)

    ids = trf_t.quat.identity()[None].repeat(J, 1).cuda(quats.device)  # (1, 4) -> (J, 4)
    ids = trf_t.quat.mul(pre_quats[:, :], ids)                         # (J, 4)
    quats = trf_t.quat.mul(pre_quats.repeat(T, 1, 1), quats)           # (T, J, 4)
                
    # 4 x 4 transformation matrix (R: identity / T: bone offsets)
    rest_mat = trf_t.quat.to_rotmat(ids)                                            # (J, 3, 3)
    rest_mat = torch.cat([rest_mat, bone_offset.unsqueeze(-1),], dim=-1)           # (J, 3, 4)
    rest_mat = torch.cat([rest_mat, torch.tensor([0, 0, 0, 1], dtype=torch.float).repeat(J, 1, 1).cuda(quats.device),], dim=-2,)  # (J, 4, 4) = (J, 3, 4) + (J, 1, 4)

    # 4 x 4 transformation matrix (R: current rot / T: bone offsets) 
    rot_mat = trf_t.quat.to_rotmat(quats)                                                                                                                        # (T, J, 3, 3)
    rot_mat = torch.cat([rot_mat, bone_offset[None, :, :, None].repeat(T, 1, 1, 1)], dim=-1)                                                       # (T, J, 3, 4) = (T, J, 3, 3) + (T, J, 3, 1)
    rot_mat = torch.cat([rot_mat, torch.tensor([0, 0, 0, 1], dtype=torch.float).repeat(T, J, 1, 1).cuda(quats.device),], dim=-2,)  # (T, J, 4, 4) = (T, J, 3, 4) + (T, J, 1, 4)

    # matrix들 복사 후 조인트 별로 잘라서 리스트 형태로 만들기 
    rest_local_trf     = rest_mat.clone()                              # (J, 4, 4)
    rot_local_trf      = rot_mat.clone()                               # (T, J, 4, 4)
    rest_local_trf_lst = list(torch.split(rest_local_trf, 1, dim=0))   # len(): J(=22), 각 요소는 (1, 4, 4)
    rot_local_trf_lst  = list(torch.split(rot_local_trf,  1, dim=1))   # len(): J(=22), 각 요소는 (T, 1, 4, 4)
    
    # world coordinate 기준으로 변경 (Glob_child = Glob_parent x Local_child)
    rest_glob_trf_lst, rot_glob_trf_lst = [rest_local_trf_lst[0]], [rot_local_trf_lst[0]]
    for i in range(1, J):
        rest_glob_trf_lst.append(torch.matmul(rest_glob_trf_lst[parents[i]][:, :, :], rest_local_trf_lst[i][:, :, :]))  # (1, 4, 4) x (1, 4, 4) -> len(): J(=22), 각 요소는 (1, 4, 4)
        rot_glob_trf_lst.append(torch.matmul(rot_glob_trf_lst[parents[i]][:, :, :], rot_local_trf_lst[i][:, :, :]))
                           
    rest_glob_trf = torch.cat(rest_glob_trf_lst, dim=0)     # (J, 4, 4)
    rot_glob_trf  = torch.cat(rot_glob_trf_lst, dim=1)      # (T, J, 4, 4)

    # rest pose에서 현재 포즈로의 변환 행렬 
    rest_glob_trf_inv = rest_glob_trf.inverse().repeat(T, 1, 1, 1)                          # (J, 4, 4) -> (T, J, 4, 4)
    rest_to_rot_trf   = torch.einsum('tjmn,tjnk->tjmk', rot_glob_trf, rest_glob_trf_inv)    # (T, J, 4, 4)

    # rest-pose일 때의 vertex position 
    verts_rest = torch.cat([rest_verts_pos, torch.ones((nov, 1), dtype=torch.float).cuda(quats.device),], dim=-1,)   # (num_verts, 3) -> (num_verts, 4)
    
    # skinning weight 적용하여 새로운 vertex position 구하기 
    verts_lbs = torch.zeros((T, nov, 4)).cuda(quats.device)                     
    for t in range(T):
        for j in range(J):
            print(skin_weights[:,j].shape)
            weight = skin_weights[:, j].unsqueeze(1)                                          # (num_verts, 1)
            tfs = rest_to_rot_trf[t, j, :, :]                                                       # (4，4)            
            verts_lbs[t, :, :] += weight * tfs.matmul(verts_rest.transpose(0, 1)).transpose(0, 1)   # (4, 4) x (4, num_verts) = (4, num_verts) -> (num_verts, 4)

    # root 복원 
    root_recover_trf = trf_t.quat.to_rotmat(trf_t.quat.identity()).cuda(quats.device)   # (3, 3)                            
    root_recover_trf = torch.cat([root_recover_trf, (root_pos).unsqueeze(-1)], dim=-1)      # (3, 4)
    root_recover_trf = torch.cat([root_recover_trf, torch.tensor([0, 0, 0, 1], dtype=torch.float)[None].cuda(quats.device),], dim=0)    # (4, 4)

    verts_lbs_root_recover = torch.zeros((verts_lbs.shape)).cuda(quats.device)
    for t in range(T):
        # (R: identity / T: current global root position)
        cur_root_pos_trf = rot_glob_trf[t, 0, :3, 3]    # (3,)
        cur_root_pos_trf = torch.cat([trf_t.quat.to_rotmat(trf_t.quat.identity()).cuda(quats.device), cur_root_pos_trf.unsqueeze(-1)], dim=-1)
        cur_root_pos_trf = torch.cat([cur_root_pos_trf, torch.tensor([0, 0, 0, 1], dtype=torch.float)[None].cuda(quats.device)], dim=0)
        
        root_recovered_trf = torch.matmul(root_recover_trf, cur_root_pos_trf.inverse())
        
        verts_lbs_root_recover[t, :, :] = root_recovered_trf.matmul(verts_lbs[t, :, :].transpose(0, 1)).transpose(0, 1)

    verts_lbs_root_recover = verts_lbs_root_recover[:, :, :3]   # (T, num_verts, 4) -> (T, num_verts, 3)

    return verts_lbs_root_recover

def linear_blend_skinning_2(selected_skel, parents, root_pos, bone_offset, quats, rest_verts_pos, skin_weights):
    '''
        root pos         (3,)
        bone_offset      (J, 3), 원점에서 시작하는 시퀀스의 glob root pos + t-pose 일 때의 bone offsets
        quats            (T, J, 4)
        rest_verts_pos   (num_vertices, 3), initial vertex positions
        skin_weights     (num_vertices, J)
    '''
    # T, J, num_of_verts 정의 
    T = quats.shape[0]
    J = bone_offset.shape[0]  # J
    nov = rest_verts_pos.shape[0]
    
    # pre-xforms 적용
    pre_xforms = torch.from_numpy(selected_skel.pre_xforms).cuda(quats.device)      # (J, 4, 4)
    pre_quats  = trf_t.xform.to_quat(pre_xforms)                                    # (J, 4)

    ids = trf_t.quat.identity()[None].repeat(J, 1).cuda(quats.device)  # (1, 4) -> (J, 4)
    ids = trf_t.quat.mul(pre_quats[:, :], ids)                         # (J, 4)
    quats = trf_t.quat.mul(pre_quats.repeat(T, 1, 1), quats)           # (T, J, 4)
                
    # 4 x 4 transformation matrix (R: identity / T: bone offsets)
    rest_mat = trf_t.quat.to_rotmat(ids)                                            # (J, 3, 3)
    rest_mat = torch.cat([rest_mat, bone_offset.unsqueeze(-1),], dim=-1)           # (J, 3, 4)
    rest_mat = torch.cat([rest_mat, torch.tensor([0, 0, 0, 1], dtype=torch.float64).repeat(J, 1, 1).cuda(quats.device),], dim=-2,)  # (J, 4, 4) = (J, 3, 4) + (J, 1, 4)

    # 4 x 4 transformation matrix (R: current rot / T: bone offsets) 
    rot_mat = trf_t.quat.to_rotmat(quats)                                                                                                                        # (T, J, 3, 3)
    rot_mat = torch.cat([rot_mat, bone_offset[None, :, :, None].repeat(T, 1, 1, 1)], dim=-1)                                                       # (T, J, 3, 4) = (T, J, 3, 3) + (T, J, 3, 1)
    rot_mat = torch.cat([rot_mat, torch.tensor([0, 0, 0, 1], dtype=torch.float64).repeat(T, J, 1, 1).cuda(quats.device),], dim=-2,)  # (T, J, 4, 4) = (T, J, 3, 4) + (T, J, 1, 4)

    # matrix들 복사 후 조인트 별로 잘라서 리스트 형태로 만들기 
    rest_local_trf     = rest_mat.clone()                              # (J, 4, 4)
    rot_local_trf      = rot_mat.clone()                               # (T, J, 4, 4)
    rest_local_trf_lst = list(torch.split(rest_local_trf, 1, dim=0))   # len(): J(=22), 각 요소는 (1, 4, 4)
    rot_local_trf_lst  = list(torch.split(rot_local_trf,  1, dim=1))   # len(): J(=22), 각 요소는 (T, 1, 4, 4)
    
    # world coordinate 기준으로 변경 (Glob_child = Glob_parent x Local_child)
    rest_glob_trf_lst, rot_glob_trf_lst = [rest_local_trf_lst[0]], [rot_local_trf_lst[0]]
    for i in range(1, J):
        rest_glob_trf_lst.append(torch.matmul(rest_glob_trf_lst[parents[i]][:, :, :], rest_local_trf_lst[i][:, :, :]))  # (1, 4, 4) x (1, 4, 4) -> len(): J(=22), 각 요소는 (1, 4, 4)
        rot_glob_trf_lst.append(torch.matmul(rot_glob_trf_lst[parents[i]][:, :, :], rot_local_trf_lst[i][:, :, :]))
                           
    rest_glob_trf = torch.cat(rest_glob_trf_lst, dim=0)     # (J, 4, 4)
    rot_glob_trf  = torch.cat(rot_glob_trf_lst, dim=1)      # (T, J, 4, 4)

    # rest pose에서 현재 포즈로의 변환 행렬 
    rest_glob_trf_inv = rest_glob_trf.inverse().repeat(T, 1, 1, 1)                          # (J, 4, 4) -> (T, J, 4, 4)
    rest_to_rot_trf   = torch.einsum('tjmn,tjnk->tjmk', rot_glob_trf, rest_glob_trf_inv)    # (T, J, 4, 4)

    # rest-pose일 때의 vertex position 
    verts_rest = torch.cat([rest_verts_pos, torch.ones((nov, 1), dtype=torch.float64).cuda(quats.device),], dim=-1,)   # (num_verts, 3) -> (num_verts, 4)
    
    # skinning weight 적용하여 새로운 vertex position 구하기 
    verts_lbs = torch.zeros((T, nov, 4),dtype=torch.float64).cuda(quats.device)                     
    for t in range(T):
        for j in range(J):
            weight = skin_weights[:, j].unsqueeze(1)                                                # (num_verts, 1)
            tfs = rest_to_rot_trf[t, j, :, :]                                                       # (4，4)            
            verts_lbs[t, :, :] += weight * tfs.matmul(verts_rest.transpose(0, 1)).transpose(0, 1)   # (4, 4) x (4, num_verts) = (4, num_verts) -> (num_verts, 4)

    # root 복원 
    root_pos = torch.from_numpy(root_pos).cuda(quats.device)
    root_recover_trf = trf_t.quat.to_rotmat(trf_t.quat.identity()).cuda(quats.device)   # (3, 3)                            
    root_recover_trf = torch.cat([root_recover_trf, (root_pos).unsqueeze(-1)], dim=-1)      # (3, 4)
    root_recover_trf = torch.cat([root_recover_trf, torch.tensor([0, 0, 0, 1], dtype=torch.float64)[None].cuda(quats.device),], dim=0)    # (4, 4)

    verts_lbs_root_recover = torch.zeros((verts_lbs.shape)).cuda(quats.device)
    for t in range(T):
        # (R: identity / T: current global root position)
        cur_root_pos_trf = rot_glob_trf[t, 0, :3, 3]    # (3,)
        cur_root_pos_trf = torch.cat([trf_t.quat.to_rotmat(trf_t.quat.identity()).cuda(quats.device), cur_root_pos_trf.unsqueeze(-1)], dim=-1)
        cur_root_pos_trf = torch.cat([cur_root_pos_trf, torch.tensor([0, 0, 0, 1], dtype=torch.float64)[None].cuda(quats.device)], dim=0)
        
        root_recovered_trf = torch.matmul(root_recover_trf, cur_root_pos_trf.inverse())
        
        verts_lbs_root_recover[t, :, :] = root_recovered_trf.matmul(verts_lbs[t, :, :].transpose(0, 1)).transpose(0, 1)

    verts_lbs_root_recover = verts_lbs_root_recover[:, :, :3]   # (T, num_verts, 4) -> (T, num_verts, 3)
    

    return verts_lbs_root_recover