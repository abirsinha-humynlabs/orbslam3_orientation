"""Synthetic end-to-end test: known head motion -> ORB-SLAM3-style multi-map output -> process().

The truth is a walking head (gravity-aligned world, z up). Its IMU readings are generated
exactly. The trajectory is cut into maps the way ORB-SLAM3 does it: each new map starts at
(0,0,0) with a random heading, and rotations are stored as world <- raw camera.
process() must give back one continuous trajectory in the A.2 operator frame.

  python -m pytest tests/  (or: python tests/test_stitch.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from orbslam3_orientation.core import Config, process            # noqa: E402
from orbslam3_orientation.geometry import exp_so3, rot_z          # noqa: E402
from orbslam3_orientation.io import Segment                       # noqa: E402

G = 9.80665


def make_truth(dur=60.0, imu_hz=200.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 1 / imu_hz)
    yaw = 0.6 * np.sin(0.21 * t) + 0.3 * np.sin(1.3 * t)          # turning head
    pitch = -0.6 + 0.15 * np.sin(0.9 * t)                         # mostly looking down
    roll = 0.05 * np.sin(1.7 * t)
    R = np.stack([rot_z(a) @ exp_so3([0, b, 0]) @ exp_so3([c, 0, 0]) for a, b, c in zip(yaw, pitch, roll)])
    p = np.stack([0.8 * t, 1.5 * np.sin(0.15 * t), 0.03 * np.sin(2 * np.pi * 1.8 * t)], 1)   # walking with bob
    dt = 1 / imu_hz
    w_body = np.stack([np.linalg.solve(np.eye(3), _logm(R[k].T @ R[k + 1]) / dt) for k in range(len(t) - 1)])
    w_body = np.vstack([w_body, w_body[-1]])
    acc_w = np.gradient(np.gradient(p, dt, axis=0), dt, axis=0)
    f_body = np.einsum("nji,nj->ni", R, acc_w + np.array([0, 0, G]))   # specific force in body frame
    gyr = w_body + np.array([0.004, -0.003, 0.002]) + 0.002 * rng.standard_normal(w_body.shape)
    acc = f_body + 0.02 * rng.standard_normal(f_body.shape)
    return t, R, p, gyr, acc


def _logm(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (th / np.sin(th) if th > 1e-9 else 1.0)


def make_segment(cuts=((20.0, 0.13), (40.0, 1.5)), seed=0):
    t_imu, R_imu, p_imu, gyr, acc = make_truth(seed=seed)
    rng = np.random.default_rng(seed + 1)
    R_raw = exp_so3([0.3, -1.2, 0.9])                       # cam0 <- imu (arbitrary mounting)
    R1 = exp_so3([0.02, -0.05, 0.03])                       # raw -> rectified
    T_raw = np.eye(4); T_raw[:3, :3] = R_raw; T_raw[:3, 3] = [0.03, -0.01, 0.02]
    T_rect = T_raw.copy(); T_rect[:3, :3] = R1 @ R_raw; T_rect[:3, 3] = R1 @ T_raw[:3, 3]
    calib = {"imu": {"T_cam0_imu": T_raw.tolist()}, "rectified_extrinsics": {"T_cam0rect_imu": T_rect.tolist()}}
    tf = np.arange(0.0, 60.0, 1 / 30.0)                     # camera frames; IMU clock = camera + offset
    off = 0.011
    idx = np.searchsorted(t_imu, tf + off).clip(0, len(t_imu) - 1)
    Rt, pt = R_imu[idx], p_imu[idx]
    mid = np.zeros(len(tf), int)
    keep = np.ones(len(tf), bool)
    for k, (tc, gap) in enumerate(cuts, 1):
        mid[tf >= tc] = k
        keep &= ~((tf >= tc - gap) & (tf < tc))             # no poses inside the gap
    R_store, p_store = np.empty_like(Rt), np.empty_like(pt)
    for k in np.unique(mid):
        s = mid == k
        first = np.where(s & keep)[0][0]
        psi = rng.uniform(-np.pi, np.pi) if k else 0.0
        Rz = rot_z(psi)
        R_store[s] = np.einsum("ij,njk,lk->nil", Rz, Rt[s], R_raw)      # world_k <- raw cam0
        p_store[s] = (Rz @ (pt[s] - pt[first]).T).T
    seg = Segment(name="synthetic", t=tf[keep], p=p_store[keep], R_w_cam0=R_store[keep], map_id=mid[keep],
                  t_imu=t_imu, acc=acc, gyr=gyr, calib=calib, imu_offset_s=off, offset_source="test")
    truth = dict(t=tf[keep], R_wi=Rt[keep], p_wi=pt[keep], R_rect=R1 @ R_raw, c_imu=-(R1 @ R_raw).T @ (R1 @ T_raw[:3, 3]))
    return seg, truth


def test_stitch_and_axes():
    seg, tr = make_segment()
    res = process(seg, Config(speed_p99_max=5.0))
    assert res["ok"] and len(res["gaps"]) == 2, res["alarms"]
    assert res["gravity_check_deg"] < 0.5
    # compare against truth expressed in the same operator world: fix the world by the first pose
    R_oh = res["R_world_head"]
    R_true_h = np.einsum("nij,kj->nik", tr["R_wi"], tr["R_rect"])
    D0 = R_oh[0] @ R_true_h[0].T                              # operator world <- true world
    err = [np.degrees(np.arccos(np.clip((np.trace(R_oh[i] @ (D0 @ R_true_h[i]).T) - 1) / 2, -1, 1)))
           for i in range(len(R_oh))]
    assert np.max(err) < 1.0, f"orientation error after stitching: max {np.max(err):.2f} deg"
    # +Y is gravity-down: true world z (up) maps to operator -Y
    up = D0 @ np.array([0, 0, 1.0])
    assert np.degrees(np.arccos(-up[1])) < 0.5, f"true up is {np.degrees(np.arccos(-up[1])):.2f} deg off -Y"
    # position: continuous across the 0.13 s gap, within constant-velocity error across 1.5 s
    p_true_h = tr["p_wi"] + np.einsum("nij,j->ni", tr["R_wi"], tr["c_imu"])
    p_true = (D0 @ (p_true_h - p_true_h[0]).T).T
    perr = np.linalg.norm(res["position"] - p_true, axis=1)
    assert perr.max() < 0.5, f"position error max {perr.max():.2f} m"
    # head axes: +Z forward is roughly horizontal-forward at the start, +Y has a downward component
    assert res["R_world_head"][0][1, 1] > 0                  # head +Y points down-ish (world +Y)
    return np.max(err), perr.max()


if __name__ == "__main__":
    e, pe = test_stitch_and_axes()
    print(f"ok: max orientation error {e:.3f} deg, max position error {pe:.3f} m")
