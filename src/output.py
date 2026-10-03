"""Write a process() result: poses.csv, poses.npz and orientation.json."""
from __future__ import annotations

import json
import os

import numpy as np

ARRAYS = ("t_camera", "t_imu", "map_id", "position", "R_world_head", "R_world_imu", "q_world_head", "q_world_imu")


def save(result: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    meta = {k: v for k, v in result.items() if k not in ARRAYS}
    paths = {"orientation.json": os.path.join(out_dir, "orientation.json")}
    if result.get("ok"):
        n = len(result["t_camera"])
        meta["n_poses"] = n
        cols = np.column_stack([result["t_camera"], result["t_imu"], result["map_id"], result["position"],
                                result["q_world_head"], result["q_world_imu"]])
        hdr = ("t_camera_s,t_imu_s,map_id,x_m,y_m,z_m,"
               "head_qx,head_qy,head_qz,head_qw,imu_qx,imu_qy,imu_qz,imu_qw")
        paths["poses.csv"] = os.path.join(out_dir, "poses.csv")
        np.savetxt(paths["poses.csv"], cols, delimiter=",", header=hdr, comments="",
                   fmt=["%.6f", "%.6f", "%d"] + ["%.6f"] * 3 + ["%.8f"] * 8)
        paths["poses.npz"] = os.path.join(out_dir, "poses.npz")
        np.savez_compressed(paths["poses.npz"], **{k: result[k] for k in ARRAYS})
    with open(paths["orientation.json"], "w") as fh:
        json.dump(meta, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return paths
