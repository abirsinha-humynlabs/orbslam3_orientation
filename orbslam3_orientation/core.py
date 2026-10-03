"""Post-process ORB-SLAM3 output into one continuous head trajectory per video segment,
in Opeth's operator-perspective axis convention, without changing ORB-SLAM3.

What ORB-SLAM3 (orbslam3 stage, make_deliverables_multi.py) writes, verified on real runs:
  * rotation  = world <- RAW left camera (cam0), although the run is labelled pose_frame "imu"
  * position  = IMU origin
  * world     = gravity-aligned, +z up, right-handed; x/y heading arbitrary
  * every atlas map ("segment" in trajectory.npz) restarts at (0,0,0) with a NEW heading

Steps (see process()):
  1. per map: world<-IMU = R_npz @ T_cam0_imu, frame guard, re-level to gravity, health check
  2. stitch kept maps in time order: heading across each gap from the bias-corrected gyro,
     position from constant velocity (short gaps) or a reference trajectory (long gaps)
  3. express everything in the operator world of Opeth A.2:
        +X right, +Y down (gravity), +Z forward (operator's facing direction at the first pose)
     with origin at the head position at the first pose; head frame = rectified left camera
     (whose axes are already +X right, +Y down, +Z forward)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .geometry import (integrate_gyro, interp3, level, log_so3, mat_to_quat, rot_z, wrap,
                       yaw_between)

G = 9.80665


@dataclass
class Config:
    min_map_s: float = 3.0            # maps shorter than this are dropped
    frame_tilt_max_deg: float = 0.5   # process(): gravity must read this close to vertical with T_cam0_imu
    frame_exact_max_deg: float = 1e-4 # export convention: the exporter levels with the same data, so the
                                      # matching extrinsic gives a residual of exactly 0 (< 1e-6 on 133 maps;
                                      # the other extrinsic >= 0.0045 deg)
    rate_rel_max: float = 0.25        # reorient: 3-axis rate residual RMS / gyro rate RMS (scale-free; an
                                      # absolute deg/s limit flags fast but good motion). Position-healthy
                                      # maps median 0.06, p97 ~0.25; diverged median 0.36
    rate_corr3_min: float = 0.95      # reorient: correlation of the 3-axis rates (healthy p5 0.98)
    grav_window_s: float = 10.0       # reorient: windowed gravity residual, window length
    grav_p90_max_deg: float = 2.0     # reorient: p90 over windows (healthy p99 1.7; diverged median 2.1)
    speed_p99_max: float = 3.0        # m/s at 10 Hz (walking head)
    speed_max_max: float = 15.0       # m/s at 10 Hz (allows brief fast motion / riding)
    gyro_corr_min: float = 0.90       # rotation-rate magnitude vs gyroscope
    bias_window_s: float = 10.0       # gyro bias estimated over this much trajectory before a gap
    vel_window_s: float = 0.5         # velocity for constant-velocity bridging
    short_gap_s: float = 2.0          # gaps up to this: position by constant velocity
    ref_align_s: float = 20.0         # reference alignment window before the gap
    lag_search_s: float = 0.4         # reference-trajectory clock search range
    bridge_skip_s: float = 0.5        # reorient: take the source pose this far inside the map (poses at a tracking loss are often off)
    bridge_window_s: float = 1.5      # reorient: average the heading over this much of the next map
    bridge_tilt_max_deg: float = 5.0  # reorient: a hand-off whose predicted tilt disagrees more than this is unreliable


# ------------------------------------------------------------------ helpers

def _rates(t, R):
    """Body rotation rate (rad/s, body frame) between consecutive poses."""
    dt = np.diff(t)
    w = log_so3(np.einsum("nji,njk->nik", R[:-1], R[1:])) / np.maximum(dt, 1e-6)[:, None]
    return w, dt


def _gyro_mean_rate(t_imu, gyr, a, b):
    """Mean gyro vector over each interval [a_i, b_i] (IMU clock)."""
    C = np.zeros_like(gyr)
    C[1:] = np.cumsum(0.5 * (gyr[1:] + gyr[:-1]) * np.diff(t_imu)[:, None], axis=0)
    return (interp3(b, t_imu, C) - interp3(a, t_imu, C)) / np.maximum(b - a, 1e-6)[:, None]


def _speeds_10hz(t, p, max_dt=0.25):
    out = []
    brk = np.where(np.diff(t) > max_dt)[0]
    for idx in np.split(np.arange(len(t)), brk + 1):
        if len(idx) < 3 or t[idx[-1]] - t[idx[0]] < 1.0:
            continue
        g = np.arange(t[idx[0]], t[idx[-1]], 0.1)
        P = interp3(g, t[idx], p[idx])
        out.append(np.linalg.norm(np.diff(P, axis=0), axis=1) / 0.1)
    return np.concatenate(out) if out else np.zeros(0)


def _tilt_deg(v):
    v = np.asarray(v) / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(v[2], -1, 1))))


def _lag(t_traj, R, t_imu, gyr, search):
    """Lag L such that t_traj + L is on the IMU clock (max rotation-rate correlation)."""
    w, dt = _rates(t_traj, R)
    ok = dt < 0.25
    wt, a, b = np.linalg.norm(w[ok], axis=1), t_traj[:-1][ok], t_traj[1:][ok]
    best = (0.0, -1.0)
    for L in np.arange(-search, search + 1e-9, 0.005):
        m = (a + L > t_imu[0]) & (b + L < t_imu[-1])
        if m.sum() < 30:
            continue
        wi = np.linalg.norm(_gyro_mean_rate(t_imu, gyr, a[m] + L, b[m] + L), axis=1)
        if np.std(wt[m]) > 1e-9 and np.std(wi) > 1e-9:
            r = float(np.corrcoef(wt[m], wi)[0, 1])
            if r > best[1]:
                best = (float(L), r)
    return best


def _fit_yaw_shift(A, B):
    """yaw + translation mapping B onto A (both gravity-aligned): (R, t)."""
    am, bm = A.mean(0), B.mean(0)
    a, b = A - am, B - bm
    psi = np.arctan2(np.sum(a[:, 1] * b[:, 0] - a[:, 0] * b[:, 1]), np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    Rz = rot_z(psi)
    return Rz, am - Rz @ bm


# ------------------------------------------------------------------ main

@dataclass
class MapInfo:
    map_id: int
    n_poses: int
    t_start: float
    t_end: float
    kept: bool
    reasons: list = field(default_factory=list)
    tilt_raw_deg: float | None = None      # gravity tilt with T_cam0_imu, before re-levelling
    tilt_rect_deg: float | None = None     # same with the rectified extrinsic (diagnostic)
    export_convention: str | None = None   # raw_cam0 | rectified_cam0 | unrecognised
    speed_p99: float | None = None
    speed_max: float | None = None
    gyro_corr: float | None = None


@dataclass
class GapInfo:
    from_map: int
    to_map: int
    gap_s: float
    yaw_applied_deg: float
    tilt_residual_deg: float
    gyro_bias_rad_s: list
    position_method: str
    heading_quality: str


def _check_map(seg, sel, R_raw, R_rect, cfg):
    t, p, Rc = seg.t[sel], seg.p[sel], seg.R_w_cam0[sel]
    info = MapInfo(int(seg.map_id[sel][0]), int(sel.sum()), float(t[0]), float(t[-1]), True)
    R_wi = Rc @ R_raw
    ti = t + seg.imu_offset_s
    inside = (ti >= seg.t_imu[0]) & (ti <= seg.t_imu[-1])
    if inside.sum() < 30 or t[-1] - t[0] < cfg.min_map_s:
        info.kept = False
        info.reasons.append(f"too short ({t[-1] - t[0]:.1f} s)")
        return info, None
    # Export-convention guard. The exporter (make_deliverables_multi.load_segment) levels each map
    # with mean(R_wb @ acc) using np.interp at these same times, so with the extrinsic it really
    # used the residual is exactly 0. Exactness tells raw from rectified even when R1 barely tilts
    # gravity (a 0.5 deg tolerance cannot: 49 of 132 maps have tilt_rect < 0.5 deg).
    f_all = np.stack([np.interp(ti, seg.t_imu, seg.acc[:, k]) for k in range(3)], 1)
    up_raw = np.einsum("nij,jk,nk->ni", Rc, R_raw, f_all).mean(0)
    up_rect = np.einsum("nij,jk,nk->ni", Rc, R_rect, f_all).mean(0)
    info.tilt_raw_deg, info.tilt_rect_deg = _tilt_deg(up_raw), _tilt_deg(up_rect)
    if info.tilt_raw_deg <= cfg.frame_exact_max_deg:
        info.export_convention = "raw_cam0"
    elif info.tilt_rect_deg <= cfg.frame_exact_max_deg:
        info.export_convention = "rectified_cam0"
        R_wi = Rc @ R_rect                       # identified exactly, so use it (and report it)
    else:
        info.export_convention = "unrecognised"
        info.reasons.append(f"export convention not recognised: gravity residual {info.tilt_raw_deg:.4f} deg "
                            f"with T_cam0_imu, {info.tilt_rect_deg:.4f} deg with T_cam0rect_imu (exactly 0 expected)")
    f = interp3(ti[inside], seg.t_imu, seg.acc)
    up = np.einsum("nij,nj->ni", R_wi[inside], f).mean(0)
    if info.tilt_raw_deg > cfg.frame_tilt_max_deg and info.export_convention == "unrecognised":
        info.reasons.append(f"frame check: gravity {info.tilt_raw_deg:.2f} deg off with T_cam0_imu")
    # health
    sp = _speeds_10hz(t, p)
    if len(sp):
        info.speed_p99, info.speed_max = float(np.percentile(sp, 99)), float(sp.max())
        if info.speed_p99 > cfg.speed_p99_max:
            info.reasons.append(f"speed p99 {info.speed_p99:.2f} m/s")
        if info.speed_max > cfg.speed_max_max:
            info.reasons.append(f"speed max {info.speed_max:.2f} m/s")
    w, dt = _rates(t, R_wi)
    ok = (dt < 0.25) & (ti[:-1] > seg.t_imu[0]) & (ti[1:] < seg.t_imu[-1])
    if ok.sum() > 30:
        wi = _gyro_mean_rate(seg.t_imu, seg.gyr, ti[:-1][ok], ti[1:][ok])
        a, b = np.linalg.norm(w[ok], axis=1), np.linalg.norm(wi, axis=1)
        info.gyro_corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None
        if info.gyro_corr is not None and info.gyro_corr < cfg.gyro_corr_min:
            info.reasons.append(f"gyro correlation {info.gyro_corr:.2f}")
    info.kept = info.kept and not info.reasons
    # re-level the map to gravity (ORB-SLAM3 already does this; done again so the result
    # does not depend on that)
    Rg = level(up)
    return info, dict(t=t, p=(Rg @ p.T).T, R=np.einsum("ij,njk->nik", Rg, R_wi))


def process(seg, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    cal = seg.calib
    R_raw = np.array(cal["imu"]["T_cam0_imu"], float)[:3, :3]                     # cam0 <- imu
    T_rect = np.array(cal["rectified_extrinsics"]["T_cam0rect_imu"], float)       # rect <- imu
    R_rect, t_rect = T_rect[:3, :3], T_rect[:3, 3]
    c_imu = -R_rect.T @ t_rect                                                    # head origin in IMU frame
    alarms = list(seg.notes)

    # ---- 1. per map
    maps, kept = [], []
    for k in np.unique(seg.map_id):
        info, data = _check_map(seg, seg.map_id == k, R_raw, R_rect, cfg)
        maps.append(info)
        if info.kept:
            kept.append((info, data))
        if any("rotation convention changed" in r for r in info.reasons):
            alarms.append(f"map {k}: " + info.reasons[0])
    kept.sort(key=lambda x: x[1]["t"][0])
    if not kept:
        return dict(ok=False, segment=seg.name, maps=[asdict(m) for m in maps], gaps=[], alarms=alarms + ["no usable map"])

    # reference trajectory clock (optional)
    ref = None
    if seg.ref is not None:
        L, r = _lag(seg.ref["t"], seg.ref["R"], seg.t_imu, seg.gyr, cfg.lag_search_s)
        if r > 0.9:
            ref = dict(seg.ref, t_imu=seg.ref["t"] + L)
        else:
            alarms.append(f"reference trajectory ignored: gyro correlation {r:.2f}")

    # ---- 2. stitch
    T, P, R, M = [kept[0][1]["t"]], [kept[0][1]["p"]], [kept[0][1]["R"]], [np.full(len(kept[0][1]["t"]), kept[0][0].map_id)]
    gaps = []
    for info, d in kept[1:]:
        t_prev, p_prev, R_prev = T[-1], P[-1], R[-1]
        t0, t1 = float(t_prev[-1]), float(d["t"][0])
        gap = t1 - t0
        # gyro bias from the trajectory just before the gap
        win = t_prev >= t0 - cfg.bias_window_s
        bias = np.zeros(3)
        if t0 - t_prev[win][0] >= 2.0:
            w, dt = _rates(t_prev[win], R_prev[win])
            ok = dt < 0.25
            a_imu = t_prev[win][:-1][ok] + seg.imu_offset_s
            b_imu = t_prev[win][1:][ok] + seg.imu_offset_s
            g_mean = _gyro_mean_rate(seg.t_imu, seg.gyr, a_imu, b_imu)
            bias = (g_mean - w[ok]).mean(0)
        dR = integrate_gyro(seg.t_imu, seg.gyr, t0 + seg.imu_offset_s, t1 + seg.imu_offset_s, bias)
        R_pred = R_prev[-1] @ dR
        psi, tilt_res = yaw_between(R_pred, d["R"][0])
        Rz = rot_z(psi)
        # position
        vw = t_prev >= t0 - cfg.vel_window_s
        v = (p_prev[-1] - p_prev[vw][0]) / max(t0 - t_prev[vw][0], 1e-3)
        method = "constant velocity"
        p_pred = p_prev[-1] + v * gap
        if gap > cfg.short_gap_s:
            method = "unknown (held at last position)"
            p_pred = p_prev[-1].copy()
            if ref is not None:
                tq_prev = t_prev[t_prev >= t0 - cfg.ref_align_s]
                rt = ref["t_imu"] - seg.imu_offset_s                       # reference on the camera clock
                cover = (rt[0] <= tq_prev[0]) and (rt[-1] >= t1) and len(tq_prev) > 30
                if cover and np.max(np.diff(rt[(rt >= t0 - 1) & (rt <= t1 + 1)]), initial=0) < 0.5:
                    A = p_prev[t_prev >= t0 - cfg.ref_align_s]
                    B = interp3(tq_prev, rt, ref["p"])
                    Rr, tr = _fit_yaw_shift(A, B)
                    p_pred = Rr @ interp3(np.array([t1]), rt, ref["p"])[0] + tr
                    method = f"reference ({ref['name']})"
        quality = ("good" if gap <= 1.0 else "fair" if gap <= 5.0 else "approximate" if gap <= 30.0 else "poor")
        if tilt_res > 5.0:
            alarms.append(f"map {info.map_id}: gyro-predicted orientation leaves {tilt_res:.1f} deg of tilt; "
                          "heading across this gap is unreliable")
            quality = "poor"
        gaps.append(GapInfo(M[-1][0].item(), info.map_id, round(gap, 3), round(float(np.degrees(psi)), 2),
                            round(tilt_res, 2), [round(float(x), 5) for x in bias], method, quality))
        T.append(d["t"])
        R.append(np.einsum("ij,njk->nik", Rz, d["R"]))
        P.append((Rz @ (d["p"] - d["p"][0]).T).T + p_pred)
        M.append(np.full(len(d["t"]), info.map_id))
    t, p_wi, R_wi, mid = map(np.concatenate, (T, P, R, M))

    # ---- 3. operator world (Opeth A.2): +X right, +Y down, +Z forward
    R_wh = np.einsum("nij,kj->nik", R_wi, R_rect)                 # world <- head (rectified left camera)
    p_wh = p_wi + np.einsum("nij,j->ni", R_wi, c_imu)
    first = t <= t[0] + 1.0
    fwd = R_wh[first][:, :, 2].mean(0)
    if np.hypot(fwd[0], fwd[1]) < 0.25:                           # looking almost straight down/up
        fwd = -R_wh[first][:, :, 1].mean(0)                       # image-up points forward then
    fh = np.array([fwd[0], fwd[1], 0.0]) / np.hypot(fwd[0], fwd[1])
    Y = np.array([0.0, 0.0, -1.0])
    Z = fh
    X = np.cross(Y, Z)
    R_ow = np.stack([X, Y, Z])                                    # operator world <- ORB world
    origin = p_wh[0]
    p_o = (R_ow @ (p_wh - origin).T).T
    R_oh = np.einsum("ij,njk->nik", R_ow, R_wh)
    R_oi = np.einsum("ij,njk->nik", R_ow, R_wi)

    # self-check: mean specific force must point to -Y (up) in the operator world
    ti = t + seg.imu_offset_s
    ins = (ti >= seg.t_imu[0]) & (ti <= seg.t_imu[-1])
    fo = np.einsum("nij,nj->ni", R_oi[ins], interp3(ti[ins], seg.t_imu, seg.acc)).mean(0)
    up_err = float(np.degrees(np.arccos(np.clip(-fo[1] / np.linalg.norm(fo), -1, 1))))
    if up_err > 1.0:
        alarms.append(f"self-check: gravity {up_err:.2f} deg off -Y in the output frame")

    dropped = [m.map_id for m in maps if not m.kept]
    return dict(
        ok=True, segment=seg.name,
        t_camera=t, t_imu=t + seg.imu_offset_s, map_id=mid,
        position=p_o, R_world_head=R_oh, R_world_imu=R_oi,
        q_world_head=mat_to_quat(R_oh), q_world_imu=mat_to_quat(R_oi),
        maps=[asdict(m) for m in maps], gaps=[asdict(g) for g in gaps], alarms=alarms,
        dropped_maps=dropped, gravity_check_deg=up_err,
        imu_offset_s=seg.imu_offset_s, imu_offset_source=seg.offset_source,
        conventions=dict(
            world_frame="ego_vio_world",
            head_frame="ego_head (rectified left camera, top-left-camera)",
            axis_convention="Opeth A.2, operator perspective: +X right, +Y down (gravity), +Z forward",
            world_definition=("origin = head position at the first pose; +Y = gravity (down); "
                              "+Z = operator's horizontal facing direction over the first second; "
                              "+X = +Y x +Z (right); right-handed"),
            head_axes="+X image right, +Y image down, +Z optical axis (forward)",
            rotation_meaning="R_world_head maps head coordinates into the world (columns = head axes in world)",
            timestamps="t_camera = frame clock; t_imu = t_camera + imu_offset_s",
            maps=("ORB-SLAM3 atlas maps stitched into one world: heading across each gap from the "
                  "bias-corrected gyroscope; position by constant velocity for gaps <= "
                  f"{cfg.short_gap_s:g} s, else from the reference trajectory if one covers the gap, "
                  "else held (flagged). No poses are emitted inside gaps."),
        ),
    )
