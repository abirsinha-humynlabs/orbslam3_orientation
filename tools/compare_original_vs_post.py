#!/usr/bin/env python3
"""What does the MCAP exporter publish from ORB-SLAM3's ORIGINAL output, and from the
post-processed output? Judged by the IMU's own sensors, so the verdict does not depend on either
tool. Read-only: nothing in the monorepo or the input folders is changed.

Both folders go through the monorepo's UNMODIFIED bitrobot_to_mcap.load_orbslam3, which
publishes every pose as ego_vio_world <- ego_imu. Tests on that published orientation:

  1. body frame. Body rates from consecutive published rotations vs the gyroscope. Q is the
     rotation that best maps the gyro rates onto them (Kabsch; a constant gyro bias drops out by
     centring). Q is then the rotation between the published body frame and the real IMU:
       angle(Q)         ~ 0 if the label ego_imu is true
       angle(Q R_ci^T)  ~ 0 if the published body is really the raw left camera
                          (R_ci = calibration imu.T_cam0_imu, camera <- IMU)
     It also reports the relative rate residual when the label is taken at face value (no Q).
  2. gravity. Specific force rotated into the published world, 10 s windows per map, angle to
     A.2 up (-Y): median over all maps, and over the maps the post-process marks healthy.

  python tools/compare_original_vs_post.py --monorepo <platform-egocentric-monorepo-v2> \
      --orbslam3 <ORB-SLAM3 segment folder> --post <reorient output folder> \
      --inputs <chunking dir with imu.csv, calibration.json>

Measured on six clips (2026-10-03): the original is published 90.6-104.3 deg off the real IMU and
within 0.2-1.6 deg of the raw camera, matching the camera->IMU calibration angle to 0.4 deg; the
post-processed output is published 0.2-1.6 deg off the IMU.

Exit code: 0 if the post-processed output passes (body frame within 3 deg of the IMU, gravity
median <= 2 deg on its healthy maps, or on all maps when none is healthy), 1 otherwise.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from check_with_mcap_loader import gravity_windows, import_loader, load_published  # noqa: E402
from src.core import _gyro_mean_rate  # noqa: E402
from src.geometry import log_so3  # noqa: E402
from src.io import _read_imu, read_npz  # noqa: E402


def ang1(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def body_frame(t, R, seg, t_imu, gyr):
    """-> (Q with published body rate ~ Q @ gyro rate, relative residual taking Q = I)."""
    dt = np.diff(t)
    ok = (dt > 0) & (dt < 0.1) & (seg[1:] == seg[:-1]) & (t[:-1] > t_imu[0]) & (t[1:] < t_imu[-1])
    i = np.where(ok)[0]
    w = np.stack([log_so3(R[k].T @ R[k + 1]) for k in i]) / dt[i][:, None]
    g = _gyro_mean_rate(t_imu, gyr, t[i], t[i + 1])
    wc, gc = w - w.mean(0), g - g.mean(0)
    U, _, Vt = np.linalg.svd(wc.T @ gc)
    Q = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    rel = np.sqrt(np.mean(np.sum((wc - gc) ** 2, 1))) / np.sqrt(np.mean(np.sum(gc ** 2, 1)))
    return Q, float(rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monorepo", required=True)
    ap.add_argument("--orbslam3", required=True, help="ORB-SLAM3's original segment folder")
    ap.add_argument("--post", required=True, help="the reorient output folder for the same segment")
    ap.add_argument("--inputs", required=True)
    a = ap.parse_args()
    B = import_loader(a.monorepo)
    cal = json.load(open(os.path.join(a.inputs, "calibration.json")))
    R_ci = np.array(cal["imu"]["T_cam0_imu"], float)[:3, :3]
    t_imu, acc, gyr = _read_imu(os.path.join(a.inputs, "imu.csv"))
    seg = read_npz(os.path.join(a.post, "trajectory.npz"))["segment"]
    assert np.array_equal(seg, read_npz(os.path.join(a.orbslam3, "trajectory.npz"))["segment"]), \
        "the two folders are not the same segment"
    rep = json.load(open(os.path.join(a.post, "orientation_report.json")))
    healthy = [m["map_id"] for m in rep["maps"] if m["healthy"]]

    tmp = tempfile.mkdtemp(prefix="compare_")
    rows = {}
    for lab, folder in (("ORB-SLAM3 original", a.orbslam3), ("post-processed", a.post)):
        t, R, missing = load_published(B, folder, tmp)
        if missing:
            print(f"note: {lab} folder lacks {', '.join(missing)} (local test copy); used stubs")
        Q, rel = body_frame(t, R, seg, t_imu, gyr)
        g_all = [x for k in np.unique(seg) for x in gravity_windows(t, R, seg == k, t_imu, acc)]
        g_ok = [x for k in healthy for x in gravity_windows(t, R, seg == k, t_imu, acc)]
        rows[lab] = (ang1(Q), ang1(Q @ R_ci.T), rel,
                     float(np.median(g_all)) if g_all else None, float(np.median(g_ok)) if g_ok else None)
    shutil.rmtree(tmp, ignore_errors=True)

    def fmt(x):
        return "    -" if x is None else f"{x:6.2f}°"
    print(f"{'published ego_imu from':22s} {'vs real IMU':>11s} {'vs raw camera':>13s} "
          f"{'rate resid as ego_imu':>21s} {'gravity vs -Y: all maps / healthy maps':>40s}")
    for lab, (q, qc, rel, ga, gh) in rows.items():
        print(f"{lab:22s} {q:10.2f}° {qc:12.2f}° {rel:21.2f} {fmt(ga):>29s} / {fmt(gh)}")
    print(f"camera -> IMU rotation in the calibration: {ang1(R_ci):.2f}°   healthy maps: {healthy or 'none'}")

    q, qc = rows["ORB-SLAM3 original"][:2]
    if qc < 3:
        print(f"original: what the exporter labels ego_imu is the RAW LEFT CAMERA ({qc:.2f}° from it), "
              f"{q:.1f}° away from the real IMU")
    q, _, _, ga, gh = rows["post-processed"]
    g = gh if gh is not None else ga
    ok = q <= 3.0 and g is not None and g <= 2.0
    print(f"post-processed: published ego_imu is {q:.2f}° from the real IMU, gravity median {g:.2f}° "
          f"({'healthy maps' if gh is not None else 'all maps, none healthy'})  -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
