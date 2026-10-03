"""Reproduce what the ORB-SLAM3 exporter does to each map, for tests and validation tools.

make_deliverables_multi.load_segment (fork b8a94cc, --pose-frame imu) levels every map with
the mean accelerometer reading: Rg = align_to_gravity(mean(R_wb @ acc)), P = Rg @ p - (Rg @ p)[0],
R = Rg @ R_wb @ T_cam0_imu^T. reorient() relies on that exactness to identify the convention,
so synthetic or cut trajectories must go through the same step.
"""
from __future__ import annotations

import numpy as np

from .geometry import level


def level_like_exporter(seg, R_ext=None):
    """In place: re-level every map of seg exactly as the exporter would (R_ext = the extrinsic
    the exporter used, default the raw T_cam0_imu)."""
    R_ext = np.array(seg.calib["imu"]["T_cam0_imu"], float)[:3, :3] if R_ext is None else R_ext
    for k in np.unique(seg.map_id):
        s = seg.map_id == k
        ti = seg.t[s] + seg.imu_offset_s
        f = np.stack([np.interp(ti, seg.t_imu, seg.acc[:, i]) for i in range(3)], 1)
        Rg = level(np.einsum("nij,jk,nk->ni", seg.R_w_cam0[s], R_ext, f).mean(0))
        p = (Rg @ seg.p[s].T).T
        seg.p[s] = p - p[0]
        seg.R_w_cam0[s] = np.einsum("ij,njk->nik", Rg, seg.R_w_cam0[s])
    return seg
