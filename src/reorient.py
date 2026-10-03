"""Re-orient ORB-SLAM3's trajectory.npz into Opeth's axis convention, map by map, keeping the
file name, keys, map structure, rows and timestamps exactly as the orbslam3 stage wrote them.

No stitching: every map keeps its own origin (positions restart at 0 for each map), and no
pose is added or removed. Only the axes change, so that EVERY map uses the same frame:

  world (per map) : +X right, +Y down (gravity), +Z forward -- Opeth A.2, operator perspective.
                    +Z is the operator's horizontal facing direction over the first second of
                    the heading-anchor map (the first healthy map; the first map when it is
                    healthy). Every other map gets that SAME world heading, carried across
                    each gap by the bias-corrected gyroscope; a hand-off that fails the tilt
                    consistency check is flagged in orientation_report.json.
  rotation        : world <- rectified left camera (the camera of left_rectified.mp4), whose
                    axes are +X image-right, +Y image-down, +Z optical axis.
  position_m      : that camera's centre in the map's world, (0,0,0) at the map's first pose.

What ORB-SLAM3 wrote (verified): rotation = world <- RAW left camera although labelled "imu";
position = IMU origin; world = gravity-aligned +z up with an arbitrary heading per map.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .core import Config, _gyro_mean_rate, _rates, check_maps
from .geometry import integrate_gyro, interp3, rot_z, yaw_between

FRAME_NOTE = ("OPETH_A2 cam0_rectified (already A.2: do not re-rotate). Opeth A.2 axes: +X right, +Y down (gravity), +Z forward (operator facing at the start "
              "of the heading-anchor map = the first healthy map; the same world heading in every map, "
              "carried across map breaks by the gyroscope). rotation = world <- rectified "
              "left camera (+X image right, +Y image down, +Z optical axis). position_m = that camera's "
              "centre. EACH SEGMENT HAS ITS OWN ORIGIN (0,0,0 at its first pose); headings are shared.")
# Default deliverable ("mcap" convention): exactly what the MCAP exporter's load_orbslam3 assumes,
# so its existing fixed rotation produces Opeth A.2 with no downstream change:
#   world  gravity-aligned, Z-up: +x right, +y forward, +z up (same heading in every map)
#   rotation world <- IMU body; position IMU origin (per-map origin)
# A.2 = R_ZUP_TO_A2 @ this (-90 deg about X, = bitrobot_to_mcap.R_DOC_OKVIS).
R_ZUP_TO_A2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
FRAME_NOTE_ZUP = ("ORB_ORIENTATION imu_zup: rotation = world <- IMU, position_m = IMU origin; world gravity-aligned "
                  "+x right, +y forward, +z up, the SAME heading in every map (operator facing at the start of the "
                  "heading-anchor map, carried across map breaks by the gyroscope). Opeth A.2 = R_x(-90 deg) applied "
                  "to this (+X right, +Y down, +Z forward). EACH SEGMENT HAS ITS OWN ORIGIN.")
POSE_HEADER_ZUP = ("timestamp_s x y z r00..r22  (world <- IMU, IMU origin; world z-up, x right, y forward, same heading "
                   "in every segment; Opeth A.2 = R_x(-90deg) applied; THIS SEGMENT'S OWN ORIGIN)")
POSE_HEADER = ("timestamp_s x y z r00..r22  (Opeth A.2: +X right, +Y down, +Z forward; rotation = world <- "
               "rectified left camera; THIS SEGMENT'S OWN ORIGIN, heading shared with the other segments)")


def _rate_pairs(seg, t, R):
    """Trajectory body rates and mean gyro over the same consecutive-pose intervals."""
    w, dt = _rates(t, R)
    o = seg.imu_offset_s
    ok = (dt < 0.1) & (t[:-1] + o > seg.t_imu[0]) & (t[1:] + o < seg.t_imu[-1])
    if ok.sum() < 30:
        return None, None
    return w[ok], _gyro_mean_rate(seg.t_imu, seg.gyr, t[:-1][ok] + o, t[1:][ok] + o)


def _orientation_metrics(seg, M, bias, cfg):
    """Does this map's ORIENTATION follow the IMU? Two independent tests:
    - 3-axis rates: trajectory body rate vs (gyro - bias), RMS residual relative to the gyro's RMS
      rate, and correlation of all three components (a magnitude correlation ignores axis errors and straddles its threshold
      on diverged maps);
    - windowed gravity: mean specific force over each cfg.grav_window_s window, angle to the
      map's vertical, p90 over windows. (A whole-map mean is forced to 0 by the levelling.)"""
    out = dict(rate_rms_dps=None, rate_rel=None, rate_corr3=None, gravity_window_p90_deg=None, gravity_window_max_deg=None)
    w, g = _rate_pairs(seg, M["t"], M["R"])
    if w is not None:
        e = np.degrees(np.linalg.norm(g - bias - w, axis=1))
        out["rate_rms_dps"] = float(np.sqrt(np.mean(e ** 2)))
        out["rate_rel"] = out["rate_rms_dps"] / max(float(np.degrees(np.sqrt(np.mean(np.sum((g - bias) ** 2, 1))))), 1e-9)
        out["rate_corr3"] = float(np.corrcoef((g - bias).ravel(), w.ravel())[0, 1])
    t, R = M["t"], M["R"]
    ti = np.clip(t + seg.imu_offset_s, seg.t_imu[0], seg.t_imu[-1])
    fw = np.einsum("nij,nj->ni", R, interp3(ti, seg.t_imu, seg.acc))
    res = []
    for a in np.arange(t[0], t[-1] - cfg.grav_window_s / 2, cfg.grav_window_s):
        m = (t >= a) & (t < a + cfg.grav_window_s)
        if m.sum() > 50:
            v = fw[m].mean(0)
            res.append(float(np.degrees(np.arccos(np.clip(v[2] / np.linalg.norm(v), -1, 1)))))
    if len(res) >= 2:                      # one window = the whole-map mean, which is 0 by construction
        out["gravity_window_p90_deg"], out["gravity_window_max_deg"] = float(np.percentile(res, 90)), max(res)
    why = []
    if M["info"].export_convention == "unrecognised":
        why.append("export convention not recognised")
    if out["rate_rms_dps"] is None:
        why.append("too few poses for the rate test")
    else:
        if out["rate_rel"] > cfg.rate_rel_max:
            why.append(f"3-axis rate residual {out['rate_rel']:.2f} of the gyro rate ({out['rate_rms_dps']:.1f} deg/s)")
        if out["rate_corr3"] < cfg.rate_corr3_min:
            why.append(f"3-axis rate correlation {out['rate_corr3']:.3f}")
    if out["gravity_window_p90_deg"] is not None and out["gravity_window_p90_deg"] > cfg.grav_p90_max_deg:
        why.append(f"windowed gravity residual p90 {out['gravity_window_p90_deg']:.2f} deg")
    return out, why


def _pooled_bias(seg, maps):
    """Gyro bias for the rate test: median (gyro - body rate) over the maps whose POSITION is
    healthy (independent of the orientation test it feeds), else over all maps."""
    pos_ok = [M for M in maps if M["info"].speed_p99 is not None and not any(
        r.startswith(("speed", "too short")) for r in M["info"].reasons)]
    est = []
    for M in pos_ok or maps:
        w, g = _rate_pairs(seg, M["t"], M["R"])
        if w is not None:
            est.append(g - w)
    return np.median(np.vstack(est), 0) if est else np.zeros(3)


def _bias(seg, t, R, t_end, window):
    """Gyro bias over the trajectory window [t_end - window, t_end]; zeros if too little data."""
    sel = (t >= t_end - window) & (t <= t_end)
    if sel.sum() < 2 or t_end - t[sel][0] < 2.0:
        return np.zeros(3)
    w, dt = _rates(t[sel], R[sel])
    ok = dt < 0.25
    if ok.sum() < 20:
        return np.zeros(3)
    a = t[sel][:-1][ok] + seg.imu_offset_s
    b = t[sel][1:][ok] + seg.imu_offset_s
    return (_gyro_mean_rate(seg.t_imu, seg.gyr, a, b) - w[ok]).mean(0)


def _segment_bias(seg, maps, window=2.0):
    """Gyro bias for the whole segment: median over 2 s windows of (gyro - trajectory body rate),
    from every map whose orientation tracks the gyro (body rates do not depend on a map's
    heading, so maps can be pooled as they are). Checked against mod-slam on 28 real ORB-SLAM3
    map breaks: this beats a bias taken next to the gap (1-10 s gaps: 2.6 vs 4.4 deg median)."""
    est = []
    for M in maps if any(M["orient_ok"] for M in maps) else []:
        if not M["orient_ok"]:
            continue
        t, R = M["t"], M["R"]
        s0 = t[0]
        while s0 + window <= t[-1]:
            sel = (t >= s0) & (t <= s0 + window)
            s0 += window
            if sel.sum() <= 10:
                continue
            w, dt = _rates(t[sel], R[sel])
            ok = dt < 0.1
            if ok.sum() <= 8:
                continue
            g = _gyro_mean_rate(seg.t_imu, seg.gyr, t[sel][:-1][ok] + seg.imu_offset_s, t[sel][1:][ok] + seg.imu_offset_s)
            est.append((g - w[ok]).mean(0))
    return (np.median(est, 0), len(est)) if len(est) >= 5 else (None, len(est))


def reorient(seg, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    cal = seg.calib
    R_raw = np.array(cal["imu"]["T_cam0_imu"], float)[:3, :3]
    T_rect = np.array(cal["rectified_extrinsics"]["T_cam0rect_imu"], float)
    R_rect, t_rect = T_rect[:3, :3], T_rect[:3, 3]
    c_imu = -R_rect.T @ t_rect
    alarms = list(seg.notes)

    # ---- per map, in ORB-SLAM3's row order (by segment id, then time)
    conv, checked = check_maps(seg, R_raw, R_rect, cfg)
    maps = []
    for k, sel, info, data in checked:
        R_wi = seg.R_w_cam0[sel] @ (R_rect if conv.startswith("rectified") else R_raw)
        if data is None:                                   # too short to re-level: use ORB-SLAM3's levelling
            data = dict(t=seg.t[sel], p=seg.p[sel], R=R_wi)
        maps.append(dict(id=k, sel=sel, info=info, t=data["t"], p=data["p"], R=data["R"]))

    # ---- orientation tests (3-axis rates, windowed gravity); a failure also makes the map unhealthy
    rate_bias = _pooled_bias(seg, maps)
    for M in maps:
        M["metrics"], why = _orientation_metrics(seg, M, rate_bias, cfg)
        M["orient_ok"] = not why
        for r in why:
            if r not in M["info"].reasons:
                M["info"].reasons.append(r)
        M["info"].kept = M["info"].kept and not why

    # ---- headings: chain through maps in TIME order
    order = sorted(range(len(maps)), key=lambda i: maps[i]["t"][0])
    # heading anchor: the first healthy map (position AND orientation pass), else the first map
    # whose orientation tracks the gyro, else the first map
    anchor = next((i for i in order if maps[i]["info"].kept and maps[i]["orient_ok"]),
                  next((i for i in order if maps[i]["orient_ok"]), order[0]))
    if not maps[anchor]["orient_ok"]:
        alarms.append("no map has a trustworthy orientation; headings are ORB-SLAM3's own")
    any_healthy = any(M["info"].kept for M in maps)
    if maps[anchor]["orient_ok"] and not any_healthy:
        alarms.append("no healthy map: the heading anchor and sources are maps whose position diverged")
    if conv.endswith("_approx"):
        alarms.append(f"ORB-SLAM3 export convention identified only approximately ({conv}): the residual is not "
                      "exactly 0, so the exporter's numerics changed upstream")
    if conv.startswith("rectified"):
        alarms.append("ORB-SLAM3 export convention changed upstream: rotation is world <- RECTIFIED cam0 "
                      "(identified and handled)")
    if conv == "unrecognised":
        alarms.append("ORB-SLAM3 export convention not recognised: no map's orientation is trusted")
    yaw = {anchor: 0.0}
    bridges = []
    seg_bias, n_bias = _segment_bias(seg, maps)

    def bridge(src, dst, forward):
        """Yaw that puts map dst into map src's (already re-headed) world, through the gyroscope.
        The last/first poses around a tracking loss are often off, so the source pose is taken
        cfg.bridge_skip_s inside the source map and the yaw is the circular mean over the
        first/last cfg.bridge_window_s of the destination map."""
        Ms, Md = maps[src], maps[dst]
        Rs = np.einsum("ij,njk->nik", rot_z(yaw[src]), Ms["R"])
        ts, td = Ms["t"], Md["t"]
        if forward:
            ia = int(np.searchsorted(ts, max(ts[-1] - cfg.bridge_skip_s, ts[0])))
            ks = np.where(td <= td[0] + cfg.bridge_window_s)[0]
        else:
            ia = int(np.searchsorted(ts, min(ts[0] + cfg.bridge_skip_s, ts[-1])))
            ks = np.where(td >= td[-1] - cfg.bridge_window_s)[0]
        ia = min(ia, len(ts) - 1)
        if seg_bias is not None:
            b = seg_bias
        else:                                              # too little trusted data: next to the gap
            b = _bias(seg, ts, Rs, ts[ia], cfg.bias_window_s) if forward else \
                _bias(seg, ts, Rs, min(ts[ia] + cfg.bias_window_s, ts[-1]), cfg.bias_window_s)
        psis, tilts = [], []
        for k in ks:
            if forward:
                dR = integrate_gyro(seg.t_imu, seg.gyr, ts[ia] + seg.imu_offset_s, td[k] + seg.imu_offset_s, b)
                Rp = Rs[ia] @ dR
            else:
                dR = integrate_gyro(seg.t_imu, seg.gyr, td[k] + seg.imu_offset_s, ts[ia] + seg.imu_offset_s, b)
                Rp = Rs[ia] @ dR.T
            p_, t_ = yaw_between(Rp, Md["R"][k])
            psis.append(p_)
            tilts.append(t_)
        psis = np.array(psis)
        z = np.exp(1j * psis).mean()
        psi = float(np.angle(z))
        spread = float(np.degrees(np.median(np.abs(np.angle(np.exp(1j * (psis - psi)))))))
        span = float(abs(td[ks].mean() - ts[ia]))
        lo, hi = sorted((ts[ia], td[ks].mean()))
        si = (seg.t_imu >= lo + seg.imu_offset_s) & (seg.t_imu <= hi + seg.imu_offset_s)
        turned = float(np.degrees(np.sum(np.linalg.norm(seg.gyr[si][:-1], axis=1) * np.diff(seg.t_imu[si])))) if si.sum() > 1 else 0.0
        tilt = float(np.median(tilts))
        # measured on real map breaks vs mod-slam: <1 s ~0.15 deg; 1-10 s ~2.6 deg median (p90 12);
        # the error grows with how much the head turns while ORB-SLAM3 is lost
        q = ("good" if span <= 1.5 and turned < 90 else "fair" if span <= 5 and turned < 360
             else "approximate" if span <= 30 else "poor")
        if not forward:
            q = {"good": "fair", "fair": "approximate"}.get(q, q)
        if tilt > cfg.bridge_tilt_max_deg or spread > cfg.bridge_tilt_max_deg:
            q = "poor"
        return psi, dict(from_map=Ms["id"], to_map=Md["id"], direction="forward" if forward else "backward",
                            gap_s=round(float(td[0] - ts[-1]) if forward else float(ts[0] - td[-1]), 3),
                            bridge_s=round(span, 3), rotation_in_bridge_deg=round(turned, 1), yaw_deg=round(float(np.degrees(psi)), 2),
                            yaw_spread_deg=round(spread, 2), tilt_residual_deg=round(tilt, 2),
                            gyro_bias_rad_s=[round(float(x), 5) for x in b], quality=q,
                            note="" if Md["orient_ok"] else "this map's own orientation fails the checks")

    def place(dst, cands, forward):
        """Bridge dst from the nearest reliable map; if the hand-off fails the tilt check (one of
        the two maps is off near the gap), try the next reliable map, then keep the nearest."""
        tried = []
        for src in cands:
            psi, rec = bridge(src, dst, forward)
            tried.append((psi, rec))
            if rec["tilt_residual_deg"] <= cfg.bridge_tilt_max_deg and rec["yaw_spread_deg"] <= cfg.bridge_tilt_max_deg:
                break
        else:
            psi, rec = tried[0]
            rec["note"] = (rec["note"] + "; " if rec["note"] else "") + \
                f"hand-off fails the tilt check from all {len(tried)} candidate map(s): heading NOT reliable"
        yaw[dst] = psi
        bridges.append(rec)
        ok = maps[dst]["orient_ok"] and rec["quality"] != "poor"
        heading_ok[dst] = ok
        # only HEALTHY maps seed later hand-offs (a map whose position diverged is not a heading
        # source, however well its rates match), unless the segment has no healthy map at all
        if ok and (maps[dst]["info"].kept or not any_healthy):
            reliable.append(dst)

    reliable, heading_ok = [anchor], {anchor: maps[anchor]["orient_ok"]}
    for i in order[order.index(anchor) + 1:]:               # forward in time from the anchor
        place(i, sorted([j for j in reliable if maps[j]["t"][0] < maps[i]["t"][0]], key=lambda j: -maps[j]["t"][-1]), True)
    for i in reversed(order[:order.index(anchor)]):          # maps before the anchor, backwards
        place(i, sorted([j for j in reliable if maps[j]["t"][0] > maps[i]["t"][0]], key=lambda j: maps[j]["t"][0]), False)

    # ---- the operator world, from the anchor map's first second
    A = maps[anchor]
    R_wh_a = np.einsum("ij,njk,lk->nil", rot_z(yaw[anchor]), A["R"], R_rect)
    first = A["t"] <= A["t"][0] + 1.0
    fwd = R_wh_a[first][:, :, 2].mean(0)
    if np.hypot(fwd[0], fwd[1]) < 0.25:
        fwd = -R_wh_a[first][:, :, 1].mean(0)
    fh = np.array([fwd[0], fwd[1], 0.0]) / np.hypot(fwd[0], fwd[1])
    Y = np.array([0.0, 0.0, -1.0])
    R_ow = np.stack([np.cross(Y, fh), Y, fh])                         # A.2 world <- ORB world (z up)

    # ---- write back in the original row order
    rotation = np.empty_like(seg.R_w_cam0)
    position = np.empty_like(seg.p)
    rotation_zup = np.empty_like(seg.R_w_cam0)
    position_zup = np.empty_like(seg.p)
    R_ow_zup = R_ZUP_TO_A2.T @ R_ow                                   # Z-up operator world <- ORB world
    report_maps = []
    for i, M in enumerate(maps):
        Rz = rot_z(yaw.get(i, 0.0))
        R_wi = np.einsum("ij,njk->nik", Rz, M["R"])
        R_wh = np.einsum("nij,kj->nik", R_wi, R_rect)                 # world <- rectified camera
        p_h = (Rz @ M["p"].T).T + np.einsum("nij,j->ni", R_wi, c_imu)
        R_o = np.einsum("ij,njk->nik", R_ow, R_wh)
        p_o = (R_ow @ (p_h - p_h[0]).T).T
        # rows of this map in the ORIGINAL file order (M["t"] is the same order as seg.t[sel])
        rotation[M["sel"]] = R_o
        position[M["sel"]] = p_o
        p_i = (Rz @ M["p"].T).T                                       # IMU origin, levelled ORB world
        rotation_zup[M["sel"]] = np.einsum("ij,njk->nik", R_ow_zup, R_wi)
        position_zup[M["sel"]] = (R_ow_zup @ (p_i - p_i[0]).T).T
        d = asdict(M["info"])
        d.update(healthy=d.pop("kept"), health_issues=d.pop("reasons"), orientation_trusted=M["orient_ok"],
                 heading_anchor=(i == anchor), heading_shared_reliably=bool(heading_ok.get(i, False)),
                 heading_source=(i in reliable), yaw_applied_deg=round(float(np.degrees(yaw.get(i, 0.0))), 2),
                 **{k: (None if v is None else round(v, 4)) for k, v in M["metrics"].items()})
        report_maps.append(d)

    return dict(export_convention=conv, rate_test_bias_rad_s=[round(float(x), 5) for x in rate_bias], gyro_bias_rad_s=None if seg_bias is None else [round(float(x), 5) for x in seg_bias],
                gyro_bias_windows=n_bias, name=seg.name, rotation=rotation, position_m=position,
                rotation_imu_zup=rotation_zup, position_imu_zup=position_zup, frame_note_zup=FRAME_NOTE_ZUP, timestamp_s=seg.t, map_id=seg.map_id,
                healthy_maps=[m["map_id"] for m in report_maps if m["healthy"]],
                unhealthy_maps=[m["map_id"] for m in report_maps if not m["healthy"]], maps=report_maps, bridges=bridges, alarms=alarms, anchor_map=maps[anchor]["id"],
                imu_offset_s=seg.imu_offset_s, imu_offset_source=seg.offset_source, frame_note=FRAME_NOTE,
                conventions=dict(axis_convention="Opeth A.2: +X right, +Y down (gravity), +Z forward",
                                 world="per map: origin at the map's first pose; heading shared by all maps "
                                       "(operator facing at the start of the anchor map), carried by the gyroscope",
                                 rotation="world <- rectified left camera (columns = camera axes in world)",
                                 position_m="rectified left camera centre, per-map origin",
                                 timestamps="unchanged (camera clock); IMU clock = timestamp_s + imu_offset_s"))
