#!/usr/bin/env python3
"""Validate reorient() (per-map re-orientation, no stitching) on real data.

1. cut  : take healthy SINGLE-map ORB-SLAM3 runs, cut each into two maps the way ORB-SLAM3 does
          (gap with no poses; the second map restarts at (0,0,0) with a random heading), run
          reorient() on the cut and on the uncut run. Per map, D = cut world <- uncut world; with a
          correctly shared heading D is the same for both maps. Error = angle between the two D.
2. real : on real MULTI-map runs with a mod-slam trajectory (one continuous map, independent of
          ORB-SLAM3): the heading offset to mod-slam just before vs just after every hand-off
          between orientation-trusted maps. A wrong hand-off shows up as a jump.

Each segment dir holds inputs/, orbslam3/ and (for real) mod-slam/vio/trajectory.txt.

  python tools/validate_reorient.py cut  SEGDIR [SEGDIR ...] [--gaps 0.13,1,5,30]
  python tools/validate_reorient.py real SEGDIR [SEGDIR ...]
"""
import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from orbslam3_orientation import load_local                 # noqa: E402
from orbslam3_orientation.geometry import rot_z, yaw_between  # noqa: E402

ZUP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
from orbslam3_orientation.reorient import reorient          # noqa: E402
from orbslam3_orientation.simulate import level_like_exporter  # noqa: E402


def load(d, ref=False):
    return load_local(os.path.join(d, "orbslam3"), os.path.join(d, "inputs"),
                      os.path.join(d, "mod-slam") if ref else None, name=os.path.basename(d))


def ang(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1))))


def mean_rot(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, 0))
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def cut(dirs, gaps, rng):
    out = {g: [] for g in gaps}
    used = 0
    for d in dirs:
        try:
            seg = load(d)
        except Exception:
            continue
        if len(np.unique(seg.map_id)) != 1 or seg.t[-1] - seg.t[0] < 80:
            continue
        base = reorient(seg)
        if not base["maps"][0]["healthy"]:
            continue
        used += 1
        for g in gaps:
            tc = rng.uniform(seg.t[0] + 20, seg.t[-1] - g - 20)
            s2 = copy.copy(seg)
            keep = ~((seg.t >= tc) & (seg.t < tc + g))
            m = (seg.t >= tc + g).astype(int)
            Rz = rot_z(rng.uniform(-np.pi, np.pi))
            first = np.where(m == 1)[0][0]
            p2, R2 = seg.p.copy(), seg.R_w_cam0.copy()
            p2[m == 1] = (Rz @ (seg.p[m == 1] - seg.p[first]).T).T
            R2[m == 1] = np.einsum("ij,njk->nik", Rz, seg.R_w_cam0[m == 1])
            s2.t, s2.p, s2.R_w_cam0, s2.map_id, s2.order = seg.t[keep], p2[keep], R2[keep], m[keep], None
            level_like_exporter(s2)                     # each part levelled as the exporter would
            r = reorient(s2)
            idx = np.searchsorted(base["timestamp_s"], r["timestamp_s"])
            # D = cut world <- uncut world, per map. A correct shared heading gives the same D for
            # both maps (D itself is identity unless the cut moved the heading anchor).
            D = [mean_rot(np.einsum("nij,nkj->nik", r["rotation"][r["map_id"] == k], base["rotation"][idx[r["map_id"] == k]]))
                 for k in (0, 1)]
            rel = next(m["heading_shared_reliably"] for m in r["maps"] if m["map_id"] != r["anchor_map"])
            out[g].append((ang(D[1], D[0]), rel, r["anchor_map"]))
            if ang(D[1], D[0]) > 5 and rel:
                b = r["bridges"][0]
                print(f"  outlier: {os.path.basename(d)} gap {g} s: {ang(D[1], D[0]):.1f} deg, bridge {b}")
    print(f"cut test: {used} healthy single-map runs, one cut per gap length")
    for g, e in out.items():
        if e:
            e = np.array(e, float)
            for lab, s in (("all", e[:, 1] >= 0), ("graded reliable", e[:, 1] == 1), ("flagged", e[:, 1] == 0)):
                if s.any():
                    x = e[s, 0]
                    print(f"  gap {g:6.2f} s {lab:16s}: heading-transfer error median {np.median(x):5.2f} deg"
                          f"  p95 {np.percentile(x, 95):5.2f}  worst {x.max():6.2f}  (n={s.sum()})")


def _offset(r, seg, R_rect, idx):
    """Circular-mean heading offset between reorient()'s world and mod-slam's over rows idx."""
    tr, Rref = seg.ref["t"], seg.ref["R"]
    k = np.clip(np.searchsorted(tr, r["timestamp_s"][idx]), 0, len(tr) - 1)
    ok = np.abs(tr[k] - r["timestamp_s"][idx]) < 0.02
    if ok.sum() < 5:
        return None
    # A.2 world -> a z-up world (x'=X, y'=Z, z'=-Y); heading about z. Heading only: mod-slam's
    # body frame is the IMU to within 0.2 deg (checked by hand-eye on PLN-024).
    R_oi = np.einsum("ij,njk,kl->nil", ZUP, r["rotation"][idx][ok], R_rect)
    return np.exp(1j * np.array([yaw_between(A, B)[0] for A, B in zip(R_oi, Rref[k[ok]])])).mean()


def real(dirs):
    """At every heading hand-off between orientation-trusted maps: the heading offset to mod-slam
    over the 2 s before the hand-off vs over the 2 s after it. Equal offsets = the shared heading
    was carried across correctly. (Whole-map averages would also include ORB-SLAM3's own heading
    drift inside a map, which this post-process does not touch.)"""
    print("real multi-map runs vs mod-slam: heading jump at each hand-off (deg; 0 = carried correctly)")
    allv = []
    for d in dirs:
        try:
            seg = load(d, ref=True)
        except Exception as e:
            print("  skip", d, e)
            continue
        if len(np.unique(seg.map_id)) < 2 or seg.ref is None:
            continue
        r = reorient(seg)
        R_rect = np.array(seg.calib["rectified_extrinsics"]["T_cam0rect_imu"], float)[:3, :3]
        trusted = {m["map_id"] for m in r["maps"] if m["orientation_trusted"]}
        t, mid = r["timestamp_s"], r["map_id"]
        res = []
        for b in r["bridges"]:
            s_, d_ = b["from_map"], b["to_map"]
            if s_ not in trusted or d_ not in trusted:
                continue
            ts, td = t[mid == s_], t[mid == d_]
            if b["direction"] == "forward":
                i_s = np.where((mid == s_) & (t >= ts[-1] - 2))[0]
                i_d = np.where((mid == d_) & (t <= td[0] + 2))[0]
            else:
                i_s = np.where((mid == s_) & (t <= ts[0] + 2))[0]
                i_d = np.where((mid == d_) & (t >= td[-1] - 2))[0]
            a_, b_ = _offset(r, seg, R_rect, i_s), _offset(r, seg, R_rect, i_d)
            if a_ is None or b_ is None:
                continue
            j = round(abs(float(np.degrees(np.angle(b_ / a_)))), 2)
            rel = next(m["heading_shared_reliably"] for m in r["maps"] if m["map_id"] == d_)
            res.append((s_, d_, b["gap_s"], j, b["quality"], rel))
            allv.append((j, rel))
        if res:
            print(f"  {os.path.basename(d)[-62:]}: " + ", ".join(
                f"map {x}->{y} gap {g:.2f}s [{q}{'' if rel else ', FLAGGED'}]: {j:.2f}" for x, y, g, j, q, rel in res))
    for lab, want in (("hand-offs graded reliable", True), ("hand-offs FLAGGED unreliable", False)):
        v = [j for j, rel in allv if rel == want]
        if v:
            print(f"  {lab}: median {np.median(v):.2f} deg, p90 {np.percentile(v, 90):.2f}, max {max(v):.2f} (n={len(v)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("cut", "real"))
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--gaps", default="0.13,1,5,30")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "cut":
        cut(a.dirs, [float(x) for x in a.gaps.split(",")], np.random.default_rng(a.seed))
    else:
        real(a.dirs)
