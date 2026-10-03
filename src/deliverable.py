"""Write a reorient() result as the SAME deliverable ORB-SLAM3's stage publishes:

  trajectory.npz                 keys segment, timestamp_s, position_m, rotation, segment_files,
                                 frame_note -- same names, dtypes, row count and row order
  segments/segNN_mapK_poses.txt  one per map, same names, `t x y z r00..r22`, fmt %.9f
  segments_manifest.json         copied; per-map extents recomputed in the new axes, frame
                                 markers and a "post_process" block describe the convention
  trajectory_a2_camera.npz       sidecar: the same rows in the other convention

Conventions (convention=):
  "mcap" (default)  trajectory.npz = world <- IMU, IMU origin, world z-up (+x right, +y forward,
                    +z up), one heading for all maps. This is exactly what the MCAP exporter's
                    load_orbslam3 assumes: its fixed -90 deg about X then yields Opeth A.2, with no
                    downstream code change. Sidecar = the A.2 camera version.
  "a2-camera"       trajectory.npz = Opeth A.2 world <- rectified left camera, camera centre.
                    Sidecar = the mcap version. A reader must NOT rotate this one again.
  report.json, status.json, ...  every other file of ORB-SLAM3's folder copied unchanged, so a
                                 reader of the orbslam3 folder finds everything it expects
  orientation_report.json        NEW sidecar: conventions, per-map health (healthy maps and the
                                 flagged ones), heading bridges, gravity check, alarms

Nothing in ORB-SLAM3's own folder is modified; out_dir must be a different folder.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np

from . import __version__
from .io import read_npz
from .reorient import POSE_HEADER, POSE_HEADER_ZUP

SIDECAR = {"mcap": "trajectory_a2_camera.npz", "a2-camera": "trajectory_imu_zup.npz"}


def _json(o):
    return o.item() if hasattr(o, "item") else o.tolist() if hasattr(o, "tolist") else str(o)


def write(res: dict, seg, out_dir: str, convention: str = "mcap") -> dict:
    if convention not in SIDECAR:
        raise ValueError(f"convention must be one of {sorted(SIDECAR)}")
    src = seg.orb_dir
    if os.path.realpath(src) == os.path.realpath(out_dir):
        raise ValueError("refusing to overwrite ORB-SLAM3's own output; give a different --out")
    z = read_npz(os.path.join(src, "trajectory.npz"))
    n = len(z["timestamp_s"])
    order = seg.order if seg.order is not None else np.arange(n)
    assert np.array_equal(z["timestamp_s"][order], res["timestamp_s"]), "row order mismatch"

    def in_file_order(R, p):
        rot, pos = np.empty_like(z["rotation"]), np.empty_like(z["position_m"])
        rot[order], pos[order] = R, p
        return rot, pos

    zup = in_file_order(res["rotation_imu_zup"], res["position_imu_zup"]) + (res["frame_note_zup"], POSE_HEADER_ZUP)
    a2c = in_file_order(res["rotation"], res["position_m"]) + (res["frame_note"], POSE_HEADER)
    (rot, pos, note, header), (s_rot, s_pos, s_note, _) = (zup, a2c) if convention == "mcap" else (a2c, zup)

    os.makedirs(os.path.join(out_dir, "segments"), exist_ok=True)
    paths = {}
    out = dict(z)
    out.update(position_m=pos, rotation=rot, frame_note=np.array(note))
    paths["trajectory.npz"] = os.path.join(out_dir, "trajectory.npz")
    np.savez_compressed(paths["trajectory.npz"], **out)      # same keys, same order
    side = dict(out, position_m=s_pos, rotation=s_rot, frame_note=np.array(s_note))
    paths[SIDECAR[convention]] = os.path.join(out_dir, SIDECAR[convention])
    np.savez_compressed(paths[SIDECAR[convention]], **side)

    files = [str(f) for f in z["segment_files"]] if "segment_files" in z else []
    extents = {}
    for k in np.unique(z["segment"]):
        sel = z["segment"] == k
        fn = files[int(k)] if int(k) < len(files) else f"seg{int(k):02d}_map{int(k)}_poses.txt"
        p = os.path.join(out_dir, "segments", os.path.basename(fn))
        np.savetxt(p, np.column_stack([z["timestamp_s"][sel], pos[sel], rot[sel].reshape(-1, 9)]),
                   fmt="%.9f", header=header)
        paths[f"segments/{os.path.basename(fn)}"] = p
        extents[os.path.basename(fn)] = np.ptp(pos[sel], axis=0)

    man = os.path.join(src, "segments_manifest.json")
    if os.path.exists(man):
        m = json.load(open(man))
        for s in m.get("segments", []):
            e = extents.get(os.path.basename(s.get("file", "")))
            if e is not None:
                s.update(extent_x_m=float(e[0]), extent_y_m=float(e[1]), extent_z_m=float(e[2]))
        orig = json.load(open(man)).get("pose_frame")
        # explicit markers describing what trajectory.npz now holds
        if convention == "mcap":
            m.update(pose_frame="imu", rotation_frame="imu", position_frame="imu",
                     world_frame="gravity_zup_x_right_y_forward", axis_convention="zup_then_rx_minus90_to_opeth_a2",
                     gravity_axis="+z_up")
            how = "apply R_x(-90 deg) (bitrobot_to_mcap.R_DOC_OKVIS) to get Opeth A.2 -- exactly what load_orbslam3 does"
        else:
            m.update(pose_frame="cam0_rectified", rotation_frame="cam0_rectified", position_frame="cam0_rectified",
                     world_frame="opeth_a2", axis_convention="opeth_a2", gravity_axis="+y_down")
            how = "already Opeth A.2: do NOT apply an OKVIS/Basalt -> A.2 rotation"
        m["post_process"] = dict(tool="orbslam3_orientation", version=__version__, convention=convention,
                                 original_pose_frame_label=orig, frame_note=note, to_opeth_a2=how,
                                 sidecar=SIDECAR[convention], report="orientation_report.json")
        paths["segments_manifest.json"] = os.path.join(out_dir, "segments_manifest.json")
        json.dump(m, open(paths["segments_manifest.json"], "w"), indent=2)
    # every other file ORB-SLAM3 published (status.json, report.json, timeshift.json, ...)
    written = {"trajectory.npz", "segments_manifest.json", "orientation_report.json", *SIDECAR.values()}
    for root, _, fs in os.walk(src):
        rel = os.path.relpath(root, src)
        if rel.split(os.sep)[0] == "segments" and rel != ".":
            continue                                       # per-map pose files are rewritten above
        for f in fs:
            r = os.path.normpath(os.path.join(rel, f))
            if r in written or r.startswith("segments" + os.sep):
                continue
            os.makedirs(os.path.join(out_dir, rel), exist_ok=True)
            paths[r] = shutil.copy2(os.path.join(root, f), os.path.join(out_dir, r))

    rep = {k: v for k, v in res.items() if k not in ("rotation", "position_m", "timestamp_s", "map_id",
                                                      "rotation_imu_zup", "position_imu_zup")}
    rep.update(tool="orbslam3_orientation", version=__version__, convention=convention,
               trajectory_npz=note, sidecar={SIDECAR[convention]: s_note}, n_poses=int(n), source=os.path.abspath(src),
               map_files={int(k): (files[int(k)] if int(k) < len(files) else None) for k in np.unique(z["segment"])})
    paths["orientation_report.json"] = os.path.join(out_dir, "orientation_report.json")
    json.dump(rep, open(paths["orientation_report.json"], "w"), indent=2, default=_json)
    return paths
