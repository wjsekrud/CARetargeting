import numpy as np

from . import rotmat, aaxis, euler, ortho6d, xform

"""
Quaternion operations
"""
def quaternion_to_euler(q):
    # Quaternion 성분 추출
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return [roll, pitch, yaw]

def mul(q0, q1):
    r0, i0, j0, k0 = np.split(q0, 4, axis=-1)
    r1, i1, j1, k1 = np.split(q1, 4, axis=-1)

    res = np.concatenate([
        r0*r1 - i0*i1 - j0*j1 - k0*k1,
        r0*i1 + i0*r1 + j0*k1 - k0*j1,
        r0*j1 - i0*k1 + j0*r1 + k0*i1,
        r0*k1 + i0*j1 - j0*i1 + k0*r1
    ], axis=-1)

    return res

def mul_vec(q, v):
    t = 2.0 * np.cross(q[..., 1:], v, axis=-1)
    res = v + q[..., 0:1] * t + np.cross(q[..., 1:], t, axis=-1)
    return res

def inv(q):
    return np.concatenate([q[..., 0:1], -q[..., 1:]], axis=-1)

def identity():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

def interpolate(q_from, q_to, t):
    """
    Args:
        q_from: (..., 4)
        q_to: (..., 4)
        t: (..., t) or (t,), or just a float
    Returns:
        interpolated quaternion (..., 4, t)
    """

    # ensure t is a numpy array
    if isinstance(t, float):
        t = np.array([t], dtype=np.float32)
    t = np.zeros_like(q_from[..., 0:1]) + t # (..., t)

    # ensure unit quaternions
    q_from_ = q_from / (np.linalg.norm(q_from, axis=-1, keepdims=True) + 1e-8) # (..., 4)
    q_to_   = q_to   / (np.linalg.norm(q_to,   axis=-1, keepdims=True) + 1e-8) # (..., 4)

    # ensure positive dot product
    dot = np.sum(q_from_ * q_to_, axis=-1) # (...,)
    neg = dot < 0.0
    dot[neg] = -dot[neg]
    q_to_[neg] = -q_to_[neg]

    # omega = arccos(dot)
    linear = dot > 0.9999
    omegas = np.arccos(dot[~linear]) # (...,)
    omegas = omegas[..., None] # (..., 1)
    sin_omegas = np.sin(omegas) # (..., 1)

    # interpolation amounts
    t0 = np.empty_like(t)
    t0[linear] = 1.0 - t[linear]
    t0[~linear] = np.sin((1.0 - t[~linear]) * omegas) / sin_omegas # (..., t)

    t1 = np.empty_like(t)
    t1[linear] = t[linear]
    t1[~linear] = np.sin(t[~linear] * omegas) / sin_omegas # (..., t)
    
    # interpolate
    q_interp = t0[..., None, :] * q_from_[..., :, None] + t1[..., None, :] * q_to_[..., :, None] # (..., 4, t)
    
    return q_interp

def between_vecs(v_from, v_to):
    v_from_ = v_from / (np.linalg.norm(v_from, axis=-1, keepdims=True) + 1e-8) # (..., 3)
    v_to_   = v_to / (np.linalg.norm(v_to,   axis=-1, keepdims=True) + 1e-8)   # (..., 3)

    dot = np.sum(v_from_ * v_to_, axis=-1) # (...,)
    cross = np.cross(v_from_, v_to_)
    cross = cross / (np.linalg.norm(cross, axis=-1, keepdims=True) + 1e-8) # (..., 3)
    
    real = np.sqrt((1.0 + dot) * 0.5) # (...,)
    imag = np.sqrt((1.0 - dot) * 0.5)[..., None] * cross
    
    return np.concatenate([real[..., None], imag], axis=-1)

def fk(local_quats, root_pos, skeleton):
    """
    Attributes:
        local_quats: (..., J, 4)
        root_pos: (..., 3), global root position
        skeleton: aPyOpenGL.agl.Skeleton
    """
    pre_xforms = np.tile(skeleton.pre_xforms, local_quats.shape[:-2] + (1, 1, 1)) # (..., J, 4, 4)
    pre_quats  = xform.to_quat(pre_xforms) # (..., J, 4)
    pre_pos    = xform.to_translation(pre_xforms) # (..., J, 3)
    pre_pos[..., 0, :] = root_pos

    global_quats = [mul(pre_quats[..., 0, :], local_quats[..., 0, :])]
    global_pos = [pre_pos[..., 0, :]]

    for i in range(1, skeleton.num_joints):
        parent_idx = skeleton.parent_idx[i]
        global_quats.append(mul(mul(global_quats[parent_idx], pre_quats[..., i, :]), local_quats[..., i, :]))
        global_pos.append(mul_vec(global_quats[parent_idx], pre_pos[..., i, :]) + global_pos[parent_idx])
    
    global_quats = np.stack(global_quats, axis=-2) # (..., J, 4)
    global_pos = np.stack(global_pos, axis=-2) # (..., J, 3)

    return global_quats, global_pos

"""
Quaternion to other representations
"""
def to_aaxis(quat):
    axis, angle = np.empty_like(quat[..., 1:]), np.empty_like(quat[..., 0])

    # small angles
    length = np.sqrt(np.sum(quat[..., 1:] * quat[..., 1:], axis=-1)) # (...,)
    small_angles = length < 1e-8

    # avoid division by zero
    angle[small_angles] = 0.0
    axis[small_angles]  = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # normal case
    angle[~small_angles] = 2.0 * np.arctan2(length[~small_angles], quat[..., 0][~small_angles]) # (...,)
    axis[~small_angles]  = quat[..., 1:][~small_angles] / length[~small_angles][..., None] # (..., 3)

    # make sure angle is in [-pi, pi)
    large_angles = angle >= np.pi
    angle[large_angles] = angle[large_angles] - 2 * np.pi

    return axis * angle[..., None] # (..., 3)

def to_rotmat(quat):
    two_s = 2.0 / np.sum(quat * quat, axis=-1) # (...,)
    r, i, j, k = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    rotmat = np.stack([
        1.0 - two_s * (j*j + k*k),
        two_s * (i*j - k*r),
        two_s * (i*k + j*r),
        two_s * (i*j + k*r),
        1.0 - two_s * (i*i + k*k),
        two_s * (j*k - i*r),
        two_s * (i*k - j*r),
        two_s * (j*k + i*r),
        1.0 - two_s * (i*i + j*j)
    ], axis=-1)
    return rotmat.reshape(quat.shape[:-1] + (3, 3)) # (..., 3, 3)

def to_ortho6d(quat):
    return rotmat.to_ortho6d(to_rotmat(quat))

def to_xform(quat, translation=None):
    return rotmat.to_xform(to_rotmat(quat), translation=translation)

"""
Other representations to quaternion
"""
def from_aaxis(a):
    return aaxis.to_quat(a)

def from_euler(angles, order, radians=True):
    return euler.to_quat(angles, order, radians=radians)

def from_rotmat(r):
    return rotmat.to_quat(r)

def from_ortho6d(r6d):
    return ortho6d.to_quat(r6d)

def from_xform(x):
    return xform.to_quat(x)