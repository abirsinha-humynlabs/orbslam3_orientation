"""Check video: does the post-processed orientation hold up on the real footage?

LEFT  the rectified left video with the OPERATOR WORLD drawn into it from each frame's pose:
        * an axis triad 1 m in front of the head: X red (right), Y green (down), Z blue (forward)
        * the horizon (yellow): the world's horizontal plane through the head
      Correct output -> green Y always points to gravity-down in the image, the horizon matches
      the scene's level, and the triad stays fixed to the scene across ORB-SLAM3 map switches.
RIGHT top view of the stitched trajectory (looking down +Y: X right, Z up on screen), one colour
      per ORB-SLAM3 map; below it the RAW ORB-SLAM3 maps as written (each from its own origin and
      heading); a timeline of maps, gaps and how each gap was bridged.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

COLORS = [(235, 120, 30), (30, 140, 255), (60, 175, 50), (200, 60, 200), (40, 200, 220), (120, 120, 240), (180, 180, 40)]
X_COL, Y_COL, Z_COL = (40, 40, 230), (40, 200, 40), (230, 120, 30)       # BGR: red, green, blue


def _proj(K, Xh):
    """Head-frame points (N,3) -> pixels (N,2), with a validity mask (in front of the camera)."""
    z = Xh[:, 2]
    ok = z > 0.05
    u = K[0] * Xh[:, 0] / np.where(ok, z, 1) + K[2]
    v = K[1] * Xh[:, 1] / np.where(ok, z, 1) + K[3]
    return np.stack([u, v], 1), ok


def render(res: dict, seg, video: str, out: str, scale: float = 0.6, max_frames: int = 0,
           title="STITCHED (operator world, top view: X right, Z forward)", lines=None, flag="dropped",
           box_mask=None, no_pose="NO POSE (ORB-SLAM3 gap or dropped map)"):
    import cv2
    L = seg.calib["rectified"]["left"]
    K = np.array([L["fx"], L["fy"], L["cx"], L["cy"]], float) * scale
    cap = cv2.VideoCapture(video)
    VW, VH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    LW, LH = int(VW * scale) // 2 * 2, int(VH * scale) // 2 * 2
    PW = LH
    ft = seg.frame_t if seg.frame_t is not None else np.arange(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) / fps

    t, P, R, mid = res["t_camera"], res["position"], res["R_world_head"], res["map_id"]
    maps = sorted(set(int(m) for m in mid))
    col = {m: COLORS[i % len(COLORS)] for i, m in enumerate(sorted(set(int(x) for x in seg.map_id)))}

    # ---------- static right-panel layers
    TOP_H, RAW_H = int(LH * 0.50), int(LH * 0.30)
    TL_H = LH - TOP_H - RAW_H
    BG, INK, GRID = (250, 250, 250), (40, 40, 40), (225, 225, 225)

    def frame_box(pts, w, h, pad=24):
        lo, hi = pts.min(0), pts.max(0)
        span = max(float((hi - lo).max()), 1.0) * 1.1
        mid_ = (lo + hi) / 2
        s = (min(w, h) - 2 * pad) / span
        return lambda q: np.stack([w / 2 + (q[..., 0] - mid_[0]) * s, h / 2 - (q[..., 1] - mid_[1]) * s], -1), s

    xz = P[:, [0, 2]]                                   # top view: X right, Z up on screen
    to_top, s_top = frame_box(xz if box_mask is None or not box_mask.any() else xz[box_mask], PW, TOP_H)
    top_bg = np.full((TOP_H, PW, 3), BG, np.uint8)
    cv2.putText(top_bg, title, (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, INK, 1, cv2.LINE_AA)
    for m in maps:
        q = to_top(xz[mid == m]).astype(np.int32)
        cv2.polylines(top_bg, [q.reshape(-1, 1, 2)], False, tuple(int(255 - (255 - c) * 0.35) for c in col[m]), 2, cv2.LINE_AA)
    o = to_top(np.zeros(2)).astype(int)                 # world origin + axis legend
    cv2.arrowedLine(top_bg, tuple(o), tuple(o + [30, 0]), X_COL, 2, tipLength=0.3)
    cv2.arrowedLine(top_bg, tuple(o), tuple(o + [0, -30]), Z_COL, 2, tipLength=0.3)
    cv2.putText(top_bg, "X", tuple(o + [33, 4]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, X_COL, 1, cv2.LINE_AA)
    cv2.putText(top_bg, "Z", tuple(o + [-4, -34]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Z_COL, 1, cv2.LINE_AA)

    raw_bg = np.full((RAW_H, PW, 3), BG, np.uint8)
    cv2.line(raw_bg, (0, 0), (PW, 0), (200, 200, 200), 1)
    cv2.putText(raw_bg, "RAW ORB-SLAM3 maps (each own origin + heading)", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, INK, 1, cv2.LINE_AA)
    rp = seg.p[:, :2]
    to_raw, _ = frame_box(rp, PW, RAW_H, 22)
    for m in sorted(col):
        q = to_raw(rp[seg.map_id == m]).astype(np.int32)
        if len(q) > 1:
            cv2.polylines(raw_bg, [q.reshape(-1, 1, 2)], False, col[m], 1, cv2.LINE_AA)

    tl_bg = np.full((TL_H, PW, 3), BG, np.uint8)
    cv2.line(tl_bg, (0, 0), (PW, 0), (200, 200, 200), 1)
    t0, t1 = float(ft[0]), float(ft[-1])
    xt = lambda tt: int(30 + (tt - t0) / max(t1 - t0, 1e-6) * (PW - 40))
    for info in res["maps"]:
        y = 22 if info["kept"] else 34
        c = col[info["map_id"]] if info["kept"] else (160, 160, 160)
        cv2.rectangle(tl_bg, (xt(info["t_start"]), y), (xt(info["t_end"]), y + 8), c, -1)
        if not info["kept"]:
            cv2.putText(tl_bg, f"map {info['map_id']} {flag}", (xt(info["t_start"]), y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(tl_bg, "maps", (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.33, INK, 1, cv2.LINE_AA)
    if lines is None:
        lines = [f"gap {g['from_map']}->{g['to_map']}: {g['gap_s']:.2f}s, yaw {g['yaw_applied_deg']:+.0f}, "
                 f"pos {g['position_method'].split(' (')[0]}, {g['heading_quality']}" for g in res["gaps"]]
    for k, txt in enumerate(lines):
        cv2.putText(tl_bg, txt, (6, 64 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.35, INK, 1, cv2.LINE_AA)

    # ---------- per frame
    tmp = out + ".tmp.mp4"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (LW + PW, LH))
    ang = np.linspace(-np.pi, np.pi, 181)
    horizon_dirs = np.stack([np.sin(ang), np.zeros_like(ang), np.cos(ang)], 1)    # world horizontal plane
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(ft) or (max_frames and i >= max_frames):
            break
        img = cv2.resize(frame, (LW, LH), interpolation=cv2.INTER_AREA)
        j = int(np.clip(np.searchsorted(t, ft[i]), 0, len(t) - 1))
        if j > 0 and abs(t[j - 1] - ft[i]) < abs(t[j] - ft[i]):
            j -= 1
        have = abs(t[j] - ft[i]) < 0.05
        cv2.rectangle(img, (0, 0), (LW, 44), (0, 0, 0), -1)
        mm, ss = divmod(ft[i] - ft[0], 60)
        if have:
            Rh, ph, m = R[j], P[j], int(mid[j])
            # horizon
            hp, okh = _proj(K, (Rh.T @ horizon_dirs.T).T)
            pts = hp[okh & (np.abs(hp[:, 0]) < 4 * LW) & (np.abs(hp[:, 1]) < 4 * LH)].astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 220, 255), 2, cv2.LINE_AA)
            # triad 1 m in front of the head, along the head's forward axis
            anchor = ph + Rh[:, 2] * 1.0
            ends = anchor + 0.25 * np.eye(3)                                  # +X, +Y, +Z of the world
            pw = np.vstack([anchor, ends])
            uv, okp = _proj(K, (Rh.T @ (pw - ph).T).T)
            if okp[0]:
                a = tuple(uv[0].astype(int))
                for k, (c, lab) in enumerate(((X_COL, "X right"), (Y_COL, "Y down"), (Z_COL, "Z fwd"))):
                    if okp[k + 1]:
                        b = tuple(uv[k + 1].astype(int))
                        cv2.arrowedLine(img, a, b, (0, 0, 0), 6, cv2.LINE_AA, tipLength=0.25)
                        cv2.arrowedLine(img, a, b, c, 3, cv2.LINE_AA, tipLength=0.25)
                        cv2.putText(img, lab, (b[0] + 4, b[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(img, lab, (b[0] + 4, b[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
            status = f"map {m}   pos X {ph[0]:+6.2f}  Y {ph[1]:+6.2f}  Z {ph[2]:+6.2f} m"
            cv2.circle(img, (12, 30), 6, col[m], -1)
        else:
            status = no_pose
        cv2.putText(img, f"t={int(mm):02d}:{ss:04.1f}  {os.path.basename(res['segment'])}", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, status, (24, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        # right panel
        top = top_bg.copy()
        if have:
            for mm_ in maps:
                sel = (mid == mm_) & (t <= ft[i])
                if sel.sum() > 1:
                    q = to_top(xz[sel]).astype(np.int32)
                    cv2.polylines(top, [q.reshape(-1, 1, 2)], False, col[mm_], 2, cv2.LINE_AA)
            c0 = to_top(xz[j]).astype(int)
            f2 = np.array([R[j][0, 2], R[j][2, 2]])
            f2 = f2 / max(np.linalg.norm(f2), 1e-6)
            cv2.circle(top, tuple(c0), 6, (20, 20, 20), -1)
            cv2.arrowedLine(top, tuple(c0), (int(c0[0] + 22 * f2[0]), int(c0[1] - 22 * f2[1])), (20, 20, 20), 2, tipLength=0.4)
        tl = tl_bg.copy()
        cv2.line(tl, (xt(ft[i]), 14), (xt(ft[i]), 46), (0, 0, 0), 1)
        panel = np.vstack([top, raw_bg, tl])
        wr.write(np.hstack([img, panel]))
        i += 1
    cap.release()
    wr.release()
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "23", "-preset", "veryfast", "-movflags", "+faststart", out], capture_output=True)
        if r.returncode == 0:
            os.remove(tmp)
            return out
    shutil.move(tmp, out)
    return out


def render_maps(res: dict, seg, video: str, out: str, **kw):
    """Check video for reorient(): NO stitching. Each map is drawn from its own origin, all in the
    same A.2 axes; the left-side triad must keep pointing the same way in the scene across map
    switches (shared heading) and Y must point down in every map."""
    healthy = {m["map_id"]: m["healthy"] for m in res["maps"]}
    r = dict(t_camera=res["timestamp_s"], position=res["position_m"], R_world_head=res["rotation"],
             map_id=res["map_id"], segment=res["name"],
             maps=[dict(m, kept=m["healthy"]) for m in res["maps"]], gaps=[])
    lines = [f"map {b['from_map']}->{b['to_map']}: heading carried {b['bridge_s']:.2f}s by gyro, "
             f"yaw {b['yaw_deg']:+.0f}, {b['quality']}" for b in res["bridges"]]
    lines.append("grey = flagged map (kept in the output, fails the health check)")
    box = np.array([healthy[int(m)] for m in res["map_id"]])
    return render(r, seg, video, out, title="PER MAP (each from its own origin; shared axes X right, Z fwd)",
                  lines=lines, flag="FLAGGED", box_mask=box, no_pose="NO POSE (gap between ORB-SLAM3 maps)", **kw)
