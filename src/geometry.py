"""Small rotation helpers (numpy only). Rotations are 3x3 matrices; R_a_b maps b-frame
coordinates into frame a (p_a = R_a_b @ p_b)."""
from __future__ import annotations

import numpy as np


def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp_so3(w):
    """Rotation vector (3,) -> rotation matrix (Rodrigues)."""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    K = skew(np.asarray(w) / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def log_so3(R):
    """(N,3,3) or (3,3) rotation(s) -> rotation vector(s)."""
    R = np.asarray(R)
    one = R.ndim == 2
    R = R[None] if one else R
    v = 0.5 * np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    s = np.linalg.norm(v, axis=1)
    th = np.arccos(np.clip((np.einsum("nii->n", R) - 1.0) / 2.0, -1.0, 1.0))
    out = v * np.where(s > 1e-9, th / np.maximum(s, 1e-12), 1.0)[:, None]
    return out[0] if one else out


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def yaw_between(R_a, R_b):
    """Yaw psi (about world +z) such that rot_z(psi) @ R_b ~= R_a, plus the leftover tilt (deg)."""
    D = R_a @ R_b.T
    psi = float(np.arctan2(D[1, 0], D[0, 0]))
    tilt = float(np.degrees(np.arccos(np.clip(D[2, 2], -1.0, 1.0))))
    return psi, tilt


def level(up):
    """Smallest rotation taking unit vector `up` onto +z."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z)
    s, c = float(np.linalg.norm(v)), float(up @ z)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = skew(v)
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def mat_to_quat(R):
    """(N,3,3) -> (N,4) quaternions as x, y, z, w (w >= 0)."""
    R = np.asarray(R)
    one = R.ndim == 2
    R = R[None] if one else R
    q = np.empty((len(R), 4))
    for i, M in enumerate(R):
        t = np.trace(M)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q[i] = [(M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s, 0.25 * s]
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
            q[i] = [0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s, (M[2, 1] - M[1, 2]) / s]
        elif M[1, 1] > M[2, 2]:
            s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
            q[i] = [(M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s, (M[0, 2] - M[2, 0]) / s]
        else:
            s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
            q[i] = [(M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s, (M[1, 0] - M[0, 1]) / s]
    q *= np.where(q[:, 3:4] < 0, -1.0, 1.0)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q[0] if one else q


def interp3(tq, t, X):
    return np.stack([np.interp(tq, t, X[:, j]) for j in range(X.shape[1])], 1)


def integrate_gyro(t_imu, gyr, t0, t1, bias=np.zeros(3)):
    """Body rotation R_b0_b1 accumulated by the (bias-corrected) gyroscope from t0 to t1."""
    m = (t_imu > t0) & (t_imu < t1)
    tt = np.concatenate([[t0], t_imu[m], [t1]])
    ww = interp3(tt, t_imu, gyr) - bias
    R = np.eye(3)
    for k in range(1, len(tt)):
        R = R @ exp_so3(0.5 * (ww[k] + ww[k - 1]) * (tt[k] - tt[k - 1]))
    return R
