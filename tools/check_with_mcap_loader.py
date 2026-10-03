#!/usr/bin/env python3
"""Run the monorepo's UNMODIFIED MCAP loader (bitrobot_to_mcap.load_orbslam3) on a reorient output
folder and check what it would publish. Read-only: nothing in the monorepo is changed.

Checks, on the poses load_orbslam3 returns (ego_vio_world <- ego_imu, on the IMU clock):
  1. they equal the A.2 camera sidecar once the camera<-IMU calibration is applied (exact)
  2. gravity: mean specific force points to -Y (A.2 up) in every map, 10 s windows (p90)
  3. the camera's horizontal facing at the start of the anchor map is +Z (world forward)

  python tools/check_with_mcap_loader.py --monorepo <platform-egocentric-monorepo-v2> \
      --out <reorient output dir> --inputs <chunking dir with imu.csv, calibration.json>

If the output folder lacks status.json or segments_manifest.json (both are copied from ORB-SLAM3's
folder; local test copies may lack them), a temporary copy with stubs is used and that is reported.
The loader then takes the IMU time offset from report.json.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.geometry import interp3  # noqa: E402
from src.io import _read_imu, read_npz  # noqa: E402


def q2m(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def ang(A, B):
    return np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", A, B) - 1) / 2, -1, 1)))


def import_loader(monorepo):
    sys.path.insert(0, os.path.join(monorepo, "stereo-pipeline-v2", "mcap_export"))
    import bitrobot_to_mcap
    return bitrobot_to_mcap


def load_published(B, folder, tmp):
    """Run the unmodified load_orbslam3 on folder. -> (t on the IMU clock, R = published
    ego_vio_world <- ego_imu, files that had to be stubbed). Real ORB-SLAM3 folders (and outputs
    copied from them) carry status.json and segments_manifest.json; local test copies may not, so
    those get stubs in a temporary copy and the loader takes the time offset from report.json."""
    stubs = {"status.json": {"status": "ok"}, "segments_manifest.json": {}}
    missing = [f for f in stubs if not os.path.exists(os.path.join(folder, f))]
    src = folder
    if missing:
        src = tempfile.mkdtemp(prefix="stubbed_", dir=tmp)
        shutil.copytree(folder, src, dirs_exist_ok=True)
        for f in missing:
            json.dump(stubs[f], open(os.path.join(src, f), "w"))
    poses, run, check, tau = B.load_orbslam3(src, Path(tmp))
    return np.array([p[0] for p in poses]), np.stack([q2m(p[2]) for p in poses]), missing


def gravity_windows(t, R, sel, t_imu, acc, window=10.0):
    """Angle (deg) between the mean specific force, rotated into the published world, and A.2 up
    (-Y), per window of the poses in sel (windows need > 50 poses)."""
    f = np.einsum("nij,nj->ni", R, interp3(np.clip(t, t_imu[0], t_imu[-1]), t_imu, acc))
    out = []
    for t0 in np.arange(t[sel][0], t[sel][-1] - window / 2, window):
        m = sel & (t >= t0) & (t < t0 + window)
        if m.sum() > 50:
            v = f[m].mean(0)
            out.append(float(np.degrees(np.arccos(np.clip(-v[1] / np.linalg.norm(v), -1, 1)))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monorepo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", required=True)
    a = ap.parse_args()
    B = import_loader(a.monorepo)

    tmp = tempfile.mkdtemp(prefix="mcap_check_")
    t, R, missing = load_published(B, a.out, tmp)                 # published ego_vio_world <- ego_imu
    if missing:
        print(f"note: output lacks {', '.join(missing)} (local test copy); used stubs in a temporary copy")
    z = read_npz(os.path.join(a.out, "trajectory.npz"))
    seg = z["segment"]
    cal = json.load(open(os.path.join(a.inputs, "calibration.json")))
    R_rect = np.array(cal["rectified_extrinsics"]["T_cam0rect_imu"])[:3, :3]
    rep = json.load(open(os.path.join(a.out, "orientation_report.json")))
    ok = True

    c = read_npz(os.path.join(a.out, "trajectory_a2_camera.npz"))
    e = ang(np.einsum("nij,kj->nik", R, R_rect), c["rotation"])
    print(f"1. published pose vs A.2 camera sidecar: max {e.max():.1e} deg")
    ok &= e.max() < 1e-2

    t_imu, acc, _ = _read_imu(os.path.join(a.inputs, "imu.csv"))
    print("2. gravity in the published world, per map (angle of mean specific force to -Y, 10 s windows):")
    for k in np.unique(seg):
        res = gravity_windows(t, R, seg == k, t_imu, acc)
        info = next(x for x in rep["maps"] if x["map_id"] == int(k))
        if res:
            print(f"   map {k}: p90 {np.percentile(res, 90):5.2f} deg, max {max(res):5.2f}"
                  f"  (healthy={info['healthy']})")
            ok &= (np.percentile(res, 90) < 2.0) or not info["healthy"]
    # 3. forward = +Z: the camera's horizontal facing at the start of the anchor map points to +Z.
    #    Same rule as reorient(): the optical axis, or image-up when the camera looks nearly
    #    straight down (horizontal part of the optical axis < 0.25).
    a_id = rep["anchor_map"]
    s = seg == a_id
    Rc = np.einsum("nij,kj->nik", R[s][:30], R_rect)
    fwd = Rc[:, :, 2].mean(0)
    rule = "optical axis"
    if np.hypot(fwd[0], fwd[2]) < 0.25:
        fwd, rule = -Rc[:, :, 1].mean(0), "image-up (camera looks nearly straight down)"
    h = np.array([fwd[0], fwd[2]]) / np.hypot(fwd[0], fwd[2])
    print(f"3. horizontal facing at the start of anchor map {a_id} ({rule}): X {h[0]:+.3f}, Z {h[1]:+.3f} (expect X ~ 0, Z ~ +1)")
    ok &= abs(h[0]) < 0.05 and h[1] > 0.99
    print("PASS" if ok else "FAIL")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
