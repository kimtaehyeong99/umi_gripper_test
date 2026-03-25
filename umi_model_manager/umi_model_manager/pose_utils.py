"""
Pose conversion utilities ported from UMI codebase.

Sources:
  - detached-umi-policy/umi/common/pose_util.py
  - detached-umi-policy/diffusion_policy/common/pose_repr_util.py
"""

import numpy as np
import scipy.spatial.transform as st


# ============================================================
# Core pose utilities (from umi/common/pose_util.py)
# ============================================================

def pose_to_mat(pose):
    """Convert pose [x,y,z,rx,ry,rz] to 4x4 matrix. Supports batch."""
    pos = pose[..., :3]
    rot = st.Rotation.from_rotvec(pose[..., 3:])
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=pos.dtype)
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = rot.as_matrix()
    mat[..., 3, 3] = 1
    return mat


def mat_to_pose(mat):
    """Convert 4x4 matrix to pose [x,y,z,rx,ry,rz]. Supports batch."""
    pos = (mat[..., :3, 3].T / mat[..., 3, 3].T).T
    rot = st.Rotation.from_matrix(mat[..., :3, :3])
    shape = pos.shape[:-1]
    pose = np.zeros(shape + (6,), dtype=pos.dtype)
    pose[..., :3] = pos
    pose[..., 3:] = rot.as_rotvec()
    return pose


def _normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out


def rot6d_to_mat(d6):
    """Convert 6D rotation representation to 3x3 rotation matrix."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = _normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out


def mat_to_rot6d(mat):
    """Convert 3x3 rotation matrix to 6D representation."""
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out


def mat_to_pose10d(mat):
    """Convert 4x4 matrix to 10D pose (pos3 + rot6d)."""
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10


def pose10d_to_mat(d10):
    """Convert 10D pose to 4x4 matrix."""
    pos = d10[..., :3]
    d6 = d10[..., 3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1] + (4, 4), dtype=d10.dtype)
    out[..., :3, :3] = rotmat
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1
    return out


# ============================================================
# Pose representation conversion (from pose_repr_util.py)
# ============================================================

def convert_pose_mat_rep(pose_mat, base_pose_mat, pose_rep='abs', backward=False):
    """
    Convert between absolute and relative pose representations.

    Forward (backward=False): absolute -> relative/delta
    Backward (backward=True): relative/delta -> absolute
    """
    if not backward:
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'relative':
            return np.linalg.inv(base_pose_mat) @ pose_mat
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")
    else:
        if pose_rep == 'abs':
            return pose_mat
        elif pose_rep == 'relative':
            return base_pose_mat @ pose_mat
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")


# ============================================================
# ROS <-> numpy conversion helpers
# ============================================================

def quaternion_to_rotvec(quat_xyzw):
    """Convert quaternion [x,y,z,w] to rotation vector [rx,ry,rz]."""
    rot = st.Rotation.from_quat(quat_xyzw)
    return rot.as_rotvec().astype(np.float64)


def rotvec_to_quaternion(rotvec):
    """Convert rotation vector [rx,ry,rz] to quaternion [x,y,z,w]."""
    rot = st.Rotation.from_rotvec(rotvec)
    return rot.as_quat().astype(np.float64)  # [x,y,z,w]


def pose_stamped_to_pose6d(pose_stamped_msg):
    """Convert PoseStamped msg to numpy [x,y,z,rx,ry,rz]."""
    p = pose_stamped_msg.pose.position
    q = pose_stamped_msg.pose.orientation
    pos = np.array([p.x, p.y, p.z], dtype=np.float64)
    quat_xyzw = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    rotvec = quaternion_to_rotvec(quat_xyzw)
    return np.concatenate([pos, rotvec])


def compute_delta_pose(current_pose6d, target_pose6d):
    """
    Compute delta pose from current to target.
    Returns [dx, dy, dz, drx, dry, drz] where:
      - position delta is in base frame
      - rotation delta is the relative rotation
    """
    delta_pos = target_pose6d[:3] - current_pose6d[:3]

    current_rot = st.Rotation.from_rotvec(current_pose6d[3:])
    target_rot = st.Rotation.from_rotvec(target_pose6d[3:])
    # delta_rot * current_rot = target_rot  =>  delta_rot = target_rot * inv(current_rot)
    delta_rot = target_rot * current_rot.inv()
    delta_rotvec = delta_rot.as_rotvec()

    return np.concatenate([delta_pos, delta_rotvec])


# ============================================================
# Coordinate frame calibration (link7 <-> camera optical)
# ============================================================

def build_T_link7_cam() -> np.ndarray:
    """
    Build the 4x4 transform from arm_r_link7 frame to camera optical frame.

    From URDF:
      arm_r_link7 -> camera_r_bottom_screw_frame:
        rpy="-1.5708  1.6668  0"  xyz="0.1082 -0.021 -0.0626"
      camera_r_bottom_screw_frame -> camera_r_link:
        rpy="0 0 0"  xyz="0.01085 0.009 0.021"
      camera_r_link -> camera_optical (standard RealSense):
        rpy="-pi/2  0  -pi/2"

    Result: link7 X ≈ optical X, link7 Y = optical -Y, link7 Z ≈ optical -Z
    (with ~5.5 degree tilt)
    """
    R1 = st.Rotation.from_euler('xyz', [-1.57079632679, 1.66678943569, 0.0])
    t1 = np.array([0.108236, -0.021, -0.062552])

    R2 = st.Rotation.identity()
    t2 = np.array([0.01085, 0.009, 0.021])

    # Standard RealSense camera_link -> optical_frame
    R_optical = st.Rotation.from_euler('xyz', [-np.pi / 2, 0.0, -np.pi / 2])

    # Chain: link7 -> screw -> cam_link -> optical
    R_total = R1 * R2 * R_optical
    t_total = t1 + R1.as_matrix() @ t2

    T = np.eye(4)
    T[:3, :3] = R_total.as_matrix()
    # NO translation offset - only rotation matters for relative pose computation.
    # Including translation causes position coupling with rotation changes,
    # leading to cm-level position errors when converting back.
    T[:3, 3] = 0.0
    return T


def apply_T_link7_cam(pose6d: np.ndarray, T_link7_cam: np.ndarray) -> np.ndarray:
    """
    Convert robot EEF pose (in base_link, link7 frame) to camera optical frame.

    P_cam_in_base = P_link7_in_base @ T_link7_cam

    This makes the relative pose computation match training data conventions.
    """
    P = pose_to_mat(pose6d)
    P_cam = P @ T_link7_cam
    return mat_to_pose(P_cam)


def apply_T_cam_link7(pose6d: np.ndarray, T_link7_cam: np.ndarray) -> np.ndarray:
    """
    Convert camera optical frame pose back to link7 frame.

    P_link7_in_base = P_cam_in_base @ inv(T_link7_cam)
    """
    T_cam_link7 = np.linalg.inv(T_link7_cam)
    P = pose_to_mat(pose6d)
    P_link7 = P @ T_cam_link7
    return mat_to_pose(P_link7)
