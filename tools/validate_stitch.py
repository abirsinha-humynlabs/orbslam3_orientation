#!/usr/bin/env python3
"""Validate the post-process on real data. Two checks, both run through process() itself.

1. cut  : take healthy SINGLE-map ORB-SLAM3 runs, cut each into two maps the way ORB-SLAM3
          does it (gap with no poses, second map from (0,0,0) with a random heading), run
          process(), and compare with the uncut trajectory. Gives heading/position error vs gap.
2. real : on real MULTI-map runs that have a healthy mod-slam trajectory (one continuous map),
          align mod-slam to the stitched result on the first kept map, then report the heading
          difference on every later map. A wrong stitch shows up as a heading jump.

Each segment dir must hold inputs/ (imu.csv, calibration.json, frame_timestamps.csv),
orbslam3/ (trajectory.npz, segments_manifest.json) and optionally mod-slam/vio/trajectory.txt.

  python tools/validate_stitch.py cut  SEGDIR [SEGDIR ...] [--gaps 0.13,1,5,30]
  python tools/validate_stitch.py real SEGDIR [SEGDIR ...]
"""
import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import Config, load_local, process          # noqa: E402
from src.geometry import interp3, rot_z, yaw_between  # noqa: E402


def load(d, ref=True):
    return load_local(os.path.join(d, "orbslam3"), os.path.join(d, "inputs"),
                      os.path.join(d, "mod-slam") if ref else None, name=os.path.basename(d))


def ang(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1))))


def cut(dirs, gaps, rng):
    out = {g: ([], []) for g in gaps}
    used = 0
    for d in dirs:
        try:
            seg = load(d, ref=False)
        except Exception:
            continue
        if len(np.unique(seg.map_id)) != 1 or seg.t[-1] - seg.t[0] < 80:
            continue
        base = process(seg)
        if not base["ok"] or base["dropped_maps"]:
            continue
        used += 1
        for g in gaps:
            tc = rng.uniform(seg.t[0] + 20, seg.t[-1] - g - 10)
            s2 = copy.copy(seg)
            keep = ~((seg.t >= tc) & (seg.t < tc + g))
            m = (seg.t >= tc + g).astype(int)
            psi = rng.uniform(-np.pi, np.pi)
            Rz = rot_z(psi)
            first = np.where(m == 1)[0][0]
            p2, R2 = seg.p.copy(), seg.R_w_cam0.copy()
            p2[m == 1] = (Rz @ (seg.p[m == 1] - seg.p[first]).T).T
            R2[m == 1] = np.einsum("ij,njk->nik", Rz, seg.R_w_cam0[m == 1])
            s2.t, s2.p, s2.R_w_cam0, s2.map_id = seg.t[keep], p2[keep], R2[keep], m[keep]
            r = process(s2)
            if not r["ok"] or len(r["gaps"]) != 1:
                continue
            # base and cut results share the same first pose -> same operator world
            idx = np.searchsorted(base["t_camera"], r["t_camera"])
            after = r["map_id"] == 1
            he = [ang(r["R_world_head"][k], base["R_world_head"][idx[k]]) for k in np.where(after)[0][:30]]
            pe = np.linalg.norm(r["position"][after][:30] - base["position"][idx[after][:30]], axis=1)
            out[g][0].append(np.median(he))
            out[g][1].append(float(np.median(pe)))
    print(f"cut test: {used} healthy single-map runs, one cut per gap length")
    for g, (he, pe) in out.items():
        if he:
            he, pe = np.array(he), np.array(pe)
            print(f"  gap {g:6.2f} s: orientation error after the gap median {np.median(he):5.2f} deg  p95 {np.percentile(he, 95):5.2f}"
                  f"  |  position error median {np.median(pe):5.2f} m  p95 {np.percentile(pe, 95):5.2f}  (n={len(he)})")


def real(dirs):
    print("real multi-map runs vs mod-slam (heading difference per map after aligning on the first kept map):")
    for d in dirs:
        try:
            seg = load(d, ref=True)
        except Exception as e:
            print("  skip", d, e)
            continue
        if len(np.unique(seg.map_id)) < 2 or seg.ref is None:
            continue
        r = process(seg)
        kept = [m["map_id"] for m in r["maps"] if m["kept"]]
        if not r["ok"] or len(kept) < 2:
            print(f"  {os.path.basename(d)[-60:]}: {len(kept)} usable map(s) - nothing to stitch")
            continue
        ref = seg.ref
        # reference orientation world<-IMU on the camera clock (lag ~ 0 for mod-slam; ignore sub-frame)
        Rref = ref["R"]
        tr = ref["t"]
        diffs = []
        base_psi = None
        for m in kept:
            sel = r["map_id"] == m
            tt = r["t_camera"][sel]
            k = np.clip(np.searchsorted(tr, tt), 0, len(tr) - 1)
            psis = [yaw_between(r["R_world_imu"][sel][i], Rref[k[i]])[0] for i in range(0, sel.sum(), 10)]
            psi = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(np.degrees(psis)))))))
            if base_psi is None:
                base_psi = psi
            diffs.append((m, round(((psi - base_psi + 180) % 360) - 180, 2)))
        print(f"  {os.path.basename(d)[-60:]}: kept maps {kept}; heading vs mod-slam relative to map {kept[0]}: {diffs}")


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
