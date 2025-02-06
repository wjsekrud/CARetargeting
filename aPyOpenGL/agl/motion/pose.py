from __future__ import annotations
from typing import Union

import numpy as np
import copy

from .skeleton import Skeleton
from aPyOpenGL import transforms as trf

class Pose:
    """
    Represents a pose of a skeleton.
    It contains the local rotation matrices of each joint and the root position.

    global_xforms[i] = global_xforms[parent_idx[i]] @ pre_xform[i] @ local_rots[i]

    Attributes:
        skeleton    (Skeleton)     : The skeleton that this pose belongs to.
        local_quats (numpy.ndarray): Local rotations of each joint in quaternion.
        root_pos    (numpy.ndarray): Root positoin in world space.
    """
    def __init__(
        self,
        skeleton: Skeleton,
        local_quats: Union[np.ndarray, list[np.ndarray]] = None,
        root_pos: np.ndarray = None,
    ):
        # set attributes
        self.skeleton    = skeleton
        self.local_quats = np.stack([trf.n_quat.identity()] * skeleton.num_joints, axis=0) if local_quats is None else np.array(local_quats)
        self.root_pos    = np.zeros(3) if root_pos is None else root_pos
        
        # check shapes
        if self.skeleton.num_joints == 0:
            raise ValueError("Cannot create a pose for an empty skeleton.")
    
    def global_xforms(self):
        gq, gp = trf.n_quat.fk(self.local_quats, self.root_pos, self.skeleton)
        gr = trf.n_quat.to_rotmat(gq)
        gx = np.stack([np.identity(4, dtype=np.float32) for _ in range(self.skeleton.num_joints)], axis=0)
        gx[:, :3, :3] = gr
        gx[:, :3,  3] = gp

        return gx
    
    def skeleton_xforms(self):
        noj = self.skeleton.num_joints
        global_xforms = self.global_xforms()

        skeleton_xforms = np.stack([np.identity(4, dtype=np.float32) for _ in range(noj - 1)], axis=0)
        for i in range(1, noj):
            parent_pos = global_xforms[self.skeleton.parent_idx[i], :3, 3]
            
            target_dir = global_xforms[i, :3, 3] - parent_pos
            target_dir = target_dir / (np.linalg.norm(target_dir, axis=-1, keepdims=True) + 1e-8)

            quat = trf.n_quat.between_vecs(np.array([0, 1, 0], dtype=np.float32), target_dir)

            skeleton_xforms[i-1, :3, :3] = trf.n_quat.to_rotmat(quat)
            skeleton_xforms[i-1, :3,  3] = (parent_pos + global_xforms[i, :3, 3]) / 2

        return skeleton_xforms
    
    def remove_joint_by_name(self, joint_name):
        joint_indices = self.skeleton.remove_joint_by_name(joint_name)
        self.local_quats = np.delete(self.local_quats, joint_indices, axis=0)

    def mirror(self, pair_indices, sym_axis=None):
        res = Pose(self.skeleton, self.local_quats.copy(), self.root_pos.copy())

        # swap joint indices
        res.local_quats = res.local_quats[pair_indices]

        # mirror by symmetry axis
        if sym_axis is None:
            sym_axis = res.skeleton.find_symmetry_axis(pair_indices)
        else:
            assert sym_axis in ["x", "y", "z"], f"Invalid axis {sym_axis} for symmetry axis, must be one of ['x', 'y', 'z']"

        if sym_axis == "x":
            res.local_quats[:, (0, 1)] *= -1
            res.root_pos[0] *= -1
        elif sym_axis == "y":
            res.local_quats[:, (0, 2)] *= -1
            res.root_pos[1] *= -1
        elif sym_axis == "z":
            res.local_quats[:, (0, 3)] *= -1
            res.root_pos[2] *= -1
        
        return res

    
    @classmethod
    def from_numpy(cls, skeleton, local_quats, root_pos):
        return cls(skeleton, local_quats, root_pos)

    @classmethod
    def from_torch(cls, skeleton, local_quats, root_pos):
        return cls(skeleton, local_quats.cpu().numpy(), root_pos.cpu().numpy())

    # """ IK functions """
    # def two_bone_ik(self, base_idx, effector_idx, target_p, eps=1e-8, facing="forward"):
    #     mid_idx = self.skeleton.parent_idx[effector_idx]
    #     if self.skeleton.parent_idx[mid_idx] != base_idx:
    #         raise ValueError(f"{base_idx} and {effector_idx} are not in a two bone IK hierarchy")

    #     a = self.global_p[base_idx]
    #     b = self.global_p[mid_idx]
    #     c = self.global_p[effector_idx]

    #     global_a_R = self.global_R[base_idx]
    #     global_b_R = self.global_R[mid_idx]

    #     lab = np.linalg.norm(b - a)
    #     lcb = np.linalg.norm(b - c)
    #     lat = np.clip(np.linalg.norm(target_p - a), eps, lab + lcb - eps)

    #     ac_ab_0 = np.arccos(np.clip(np.dot(mathops.normalize(c - a), mathops.normalize(b - a)), -1, 1))
    #     ba_bc_0 = np.arccos(np.clip(np.dot(mathops.normalize(a - b), mathops.normalize(c - b)), -1, 1))
    #     ac_at_0 = np.arccos(np.clip(np.dot(mathops.normalize(c - a), mathops.normalize(target_p - a)), -1, 1))

    #     ac_ab_1 = np.arccos(np.clip((lcb*lcb - lab*lab - lat*lat) / (-2*lab*lat), -1, 1))
    #     ba_bc_1 = np.arccos(np.clip((lat*lat - lab*lab - lcb*lcb) / (-2*lab*lcb), -1, 1))

    #     axis_0 = mathops.normalize(np.cross(c - a, self.forward if facing == "forward" else -self.forward))
    #     axis_1 = mathops.normalize(np.cross(c - a, target_p - a))

    #     r0 = rotation.A_to_R(ac_ab_1 - ac_ab_0, rotation.R_inv(global_a_R) @ axis_0)
    #     r1 = rotation.A_to_R(ba_bc_1 - ba_bc_0, rotation.R_inv(global_b_R) @ axis_0)
    #     r2 = rotation.A_to_R(ac_at_0, rotation.R_inv(global_a_R) @ axis_1)

    #     self.local_R[base_idx] = self.local_R[base_idx] @ r0 @ r2
    #     self.local_R[mid_idx] = self.local_R[mid_idx] @ r1

    #     self.update()