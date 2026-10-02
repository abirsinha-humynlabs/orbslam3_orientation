"""Write a reorient() result as the SAME deliverable ORB-SLAM3's stage publishes:

  trajectory.npz                 keys segment, timestamp_s, position_m, rotation, segment_files,
                                 frame_note -- same names, dtypes, row count and row order
  segments/segNN_mapK_poses.txt  one per map, same names, `t x y z r00..r22`, fmt %.9f
  segments_manifest.json         copied; per-map extents recomputed in the new axes,
                                 "pose_frame" and a "post_process" block describe the change
  report.json                    copied unchanged
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
from .reorient import POSE_HEADER


def _json(o):
    return o.item() if hasattr(o, "item") else o.tolist() if hasattr(o, "tolist") else str(o)


def write(res: dict, seg, out_dir: str) -> dict:
    src = seg.orb_dir
    if os.path.realpath(src) == os.path.realpath(out_dir):
        raise ValueError("refusing to overwrite ORB-SLAM3's own output; give a different --out")
    z = np.load(os.path.join(src, "trajectory.npz"), allow_pickle=False)
    n = len(z["timestamp_s"])
    order = seg.order if seg.order is not None else np.arange(n)
    rot = np.empty_like(z["rotation"])
    pos = np.empty_like(z["position_m"])
    rot[order] = res["rotation"]
    pos[order] = res["position_m"]
    assert np.array_equal(z["timestamp_s"][order], res["timestamp_s"]), "row order mismatch"

    os.makedirs(os.path.join(out_dir, "segments"), exist_ok=True)
    paths = {}
    out = {k: z[k] for k in z.files}
    out.update(position_m=pos, rotation=rot, frame_note=np.array(res["frame_note"]))
    paths["trajectory.npz"] = os.path.join(out_dir, "trajectory.npz")
    np.savez_compressed(paths["trajectory.npz"], **out)      # same keys, same order

    files = [str(f) for f in z["segment_files"]] if "segment_files" in z.files else []
    extents = {}
    for k in np.unique(z["segment"]):
        sel = z["segment"] == k
        fn = files[int(k)] if int(k) < len(files) else f"seg{int(k):02d}_map{int(k)}_poses.txt"
        p = os.path.join(out_dir, "segments", os.path.basename(fn))
        np.savetxt(p, np.column_stack([z["timestamp_s"][sel], pos[sel], rot[sel].reshape(-1, 9)]),
                   fmt="%.9f", header=POSE_HEADER)
        paths[f"segments/{os.path.basename(fn)}"] = p
        extents[os.path.basename(fn)] = np.ptp(pos[sel], axis=0)

    man = os.path.join(src, "segments_manifest.json")
    if os.path.exists(man):
        m = json.load(open(man))
        for s in m.get("segments", []):
            e = extents.get(os.path.basename(s.get("file", "")))
            if e is not None:
                s.update(extent_x_m=float(e[0]), extent_y_m=float(e[1]), extent_z_m=float(e[2]))
        m["pose_frame"] = "cam0_rectified"
        m["post_process"] = dict(tool="orbslam3_orientation", version=__version__, original_pose_frame_label=json.load(open(man)).get("pose_frame"),
                                 axis_convention=res["conventions"]["axis_convention"], frame_note=res["frame_note"],
                                 report="orientation_report.json")
        paths["segments_manifest.json"] = os.path.join(out_dir, "segments_manifest.json")
        json.dump(m, open(paths["segments_manifest.json"], "w"), indent=2)
    if os.path.exists(os.path.join(src, "report.json")):
        paths["report.json"] = shutil.copy(os.path.join(src, "report.json"), os.path.join(out_dir, "report.json"))

    rep = {k: v for k, v in res.items() if k not in ("rotation", "position_m", "timestamp_s", "map_id")}
    rep.update(tool="orbslam3_orientation", version=__version__, n_poses=int(n), source=os.path.abspath(src),
               map_files={int(k): (files[int(k)] if int(k) < len(files) else None) for k in np.unique(z["segment"])})
    paths["orientation_report.json"] = os.path.join(out_dir, "orientation_report.json")
    json.dump(rep, open(paths["orientation_report.json"], "w"), indent=2, default=_json)
    return paths
