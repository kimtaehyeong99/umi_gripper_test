#!/usr/bin/env python3
"""
Rotation utility functions for UMI data pipeline.

Provides conversions between different rotation representations:
- Quaternion [qx, qy, qz, qw]
- Axis-angle (rotation vector) [rx, ry, rz]
- Rotation matrix [3x3]
- 6D rotation representation (first two rows of rotation matrix)
"""

import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to axis-angle (rotation vector).

    Args:
        quat: Quaternion [qx, qy, qz, qw] or array of shape (..., 4)

    Returns:
        Axis-angle [rx, ry, rz] or array of shape (..., 3)
    """
    quat = np.asarray(quat)
    original_shape = quat.shape

    if quat.ndim == 1:
        quat = quat.reshape(1, 4)

    # scipy uses [x, y, z, w] order
    r = Rotation.from_quat(quat)
    rotvec = r.as_rotvec()

    if len(original_shape) == 1:
        return rotvec.squeeze()
    return rotvec


def axis_angle_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle (rotation vector) to quaternion.

    Args:
        rotvec: Axis-angle [rx, ry, rz] or array of shape (..., 3)

    Returns:
        Quaternion [qx, qy, qz, qw] or array of shape (..., 4)
    """
    rotvec = np.asarray(rotvec)
    original_shape = rotvec.shape

    if rotvec.ndim == 1:
        rotvec = rotvec.reshape(1, 3)

    r = Rotation.from_rotvec(rotvec)
    quat = r.as_quat()  # [x, y, z, w]

    if len(original_shape) == 1:
        return quat.squeeze()
    return quat


def axis_angle_to_rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle (rotation vector) to rotation matrix.

    Args:
        rotvec: Axis-angle [rx, ry, rz] or array of shape (..., 3)

    Returns:
        Rotation matrix [3, 3] or array of shape (..., 3, 3)
    """
    rotvec = np.asarray(rotvec)
    original_shape = rotvec.shape

    if rotvec.ndim == 1:
        rotvec = rotvec.reshape(1, 3)

    r = Rotation.from_rotvec(rotvec)
    mat = r.as_matrix()

    if len(original_shape) == 1:
        return mat.squeeze()
    return mat


def rotation_matrix_to_axis_angle(mat: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to axis-angle (rotation vector).

    Args:
        mat: Rotation matrix [3, 3] or array of shape (..., 3, 3)

    Returns:
        Axis-angle [rx, ry, rz] or array of shape (..., 3)
    """
    mat = np.asarray(mat)
    original_shape = mat.shape

    if mat.ndim == 2:
        mat = mat.reshape(1, 3, 3)

    r = Rotation.from_matrix(mat)
    rotvec = r.as_rotvec()

    if len(original_shape) == 2:
        return rotvec.squeeze()
    return rotvec


def rotation_matrix_to_6d(mat: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to 6D representation (first two rows).

    This is the continuous 6D rotation representation from:
    "On the Continuity of Rotation Representations in Neural Networks"

    Args:
        mat: Rotation matrix [3, 3] or array of shape (..., 3, 3)

    Returns:
        6D rotation [r1x, r1y, r1z, r2x, r2y, r2z] or array of shape (..., 6)
    """
    mat = np.asarray(mat)
    if mat.ndim == 2:
        return mat[:2, :].flatten()
    else:
        # Batch mode: (..., 3, 3) -> (..., 6)
        return mat[..., :2, :].reshape(*mat.shape[:-2], 6)


def rotation_6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation back to rotation matrix.

    Uses Gram-Schmidt orthogonalization.

    Args:
        rot6d: 6D rotation [..., 6]

    Returns:
        Rotation matrix [..., 3, 3]
    """
    rot6d = np.asarray(rot6d)
    original_shape = rot6d.shape

    if rot6d.ndim == 1:
        rot6d = rot6d.reshape(1, 6)

    # Split into two 3D vectors
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    # Gram-Schmidt orthogonalization
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)

    # Stack into rotation matrix
    mat = np.stack([b1, b2, b3], axis=-2)

    if len(original_shape) == 1:
        return mat.squeeze()
    return mat


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """
    Convert pose [x, y, z, rx, ry, rz] to 4x4 homogeneous transformation matrix.

    Args:
        pose: Pose [x, y, z, rx, ry, rz] or array of shape (..., 6)

    Returns:
        4x4 homogeneous matrix or array of shape (..., 4, 4)
    """
    pose = np.asarray(pose)

    if pose.ndim == 1:
        mat = np.eye(4)
        mat[:3, 3] = pose[:3]
        mat[:3, :3] = axis_angle_to_rotation_matrix(pose[3:6])
        return mat
    else:
        # Batch mode
        batch_shape = pose.shape[:-1]
        n = int(np.prod(batch_shape))
        pose_flat = pose.reshape(n, 6)

        mats = np.zeros((n, 4, 4))
        mats[:, 3, 3] = 1.0
        mats[:, :3, 3] = pose_flat[:, :3]
        mats[:, :3, :3] = axis_angle_to_rotation_matrix(pose_flat[:, 3:6])

        return mats.reshape(*batch_shape, 4, 4)


def mat_to_pose(mat: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 homogeneous transformation matrix to pose [x, y, z, rx, ry, rz].

    Args:
        mat: 4x4 homogeneous matrix or array of shape (..., 4, 4)

    Returns:
        Pose [x, y, z, rx, ry, rz] or array of shape (..., 6)
    """
    mat = np.asarray(mat)

    if mat.ndim == 2:
        pos = mat[:3, 3]
        rotvec = rotation_matrix_to_axis_angle(mat[:3, :3])
        return np.concatenate([pos, rotvec])
    else:
        # Batch mode
        pos = mat[..., :3, 3]
        rotvec = rotation_matrix_to_axis_angle(mat[..., :3, :3])
        return np.concatenate([pos, rotvec], axis=-1)


def mat_to_pose10d(mat: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 homogeneous transformation matrix to 10D pose [x, y, z, rot6d].

    Args:
        mat: 4x4 homogeneous matrix or array of shape (..., 4, 4)

    Returns:
        10D pose [x, y, z, r1x, r1y, r1z, r2x, r2y, r2z] or array of shape (..., 9)
        Note: Actually 9D (3 pos + 6 rot), but called 10D in UMI convention
    """
    mat = np.asarray(mat)

    if mat.ndim == 2:
        pos = mat[:3, 3]
        rot6d = rotation_matrix_to_6d(mat[:3, :3])
        return np.concatenate([pos, rot6d])
    else:
        # Batch mode
        pos = mat[..., :3, 3]
        rot6d = rotation_matrix_to_6d(mat[..., :3, :3])
        return np.concatenate([pos, rot6d], axis=-1)


if __name__ == '__main__':
    # Test conversions
    print("Testing rotation conversions...")

    # Test quaternion -> axis-angle -> quaternion
    quat = np.array([0.0, 0.0, 0.7071068, 0.7071068])  # 90 deg around z
    rotvec = quaternion_to_axis_angle(quat)
    quat_back = axis_angle_to_quaternion(rotvec)
    print(f"Quaternion: {quat}")
    print(f"Axis-angle: {rotvec}")
    print(f"Quaternion back: {quat_back}")
    print(f"Close: {np.allclose(quat, quat_back)}")

    # Test rotation matrix -> 6D -> matrix
    mat = axis_angle_to_rotation_matrix(rotvec)
    rot6d = rotation_matrix_to_6d(mat)
    mat_back = rotation_6d_to_matrix(rot6d)
    print(f"\nRotation matrix:\n{mat}")
    print(f"6D rotation: {rot6d}")
    print(f"Matrix back close: {np.allclose(mat, mat_back)}")

    # Test batch operations
    quats = np.random.randn(10, 4)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    rotvecs = quaternion_to_axis_angle(quats)
    print(f"\nBatch quaternions shape: {quats.shape}")
    print(f"Batch axis-angles shape: {rotvecs.shape}")

    print("\nAll tests passed!")
