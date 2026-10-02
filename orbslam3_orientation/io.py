"""Load one video segment's ORB-SLAM3 output and its inputs, from local folders or S3.

S3 access is read-only: objects are only downloaded (get_object), never written.

Expected files
  orbslam3 dir : trajectory.npz (timestamp_s, position_m, rotation, segment),
                 segments_manifest.json and/or report.json (measured_imu_time_offset_s)
  inputs dir   : imu.csv, calibration.json, frame_timestamps.csv   (chunking/<segment>/)
  mod-slam dir : vio/trajectory.txt  (optional; only used to bridge long gaps)
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

ORB_FILES = ("trajectory.npz", "segments_manifest.json", "report.json")
INPUT_FILES = ("imu.csv", "calibration.json", "frame_timestamps.csv")
MOD_FILES = ("vio/trajectory.txt",)


@dataclass
class Segment:
    name: str
    t: np.ndarray              # (N,) ORB pose times, camera clock (s)
    p: np.ndarray              # (N,3) positions, ORB world (IMU origin)
    R_w_cam0: np.ndarray       # (N,3,3) ORB rotation: world <- RAW left camera (cam0)
    map_id: np.ndarray         # (N,) ORB atlas map index
    t_imu: np.ndarray          # (M,) IMU times (IMU clock)
    acc: np.ndarray            # (M,3) accelerometer, IMU frame (m/s^2)
    gyr: np.ndarray            # (M,3) gyroscope, IMU frame (rad/s)
    calib: dict
    imu_offset_s: float        # t_imu = t_camera + imu_offset_s
    offset_source: str
    frame_t: np.ndarray | None = None
    ref: dict | None = None    # optional reference trajectory (world <- IMU), e.g. mod-slam
    notes: list = field(default_factory=list)
    order: np.ndarray | None = None  # sorted row k = original npz row order[k]
    orb_dir: str = ""         # where trajectory.npz came from (the re-orient writer mirrors it)


def _read_imu(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    t = np.asarray(d["timestamp_s"], float)
    o = np.argsort(t)
    acc = np.stack([d["ax"], d["ay"], d["az"]], 1).astype(float)[o]
    gyr = np.stack([d["gx"], d["gy"], d["gz"]], 1).astype(float)[o]
    return t[o], acc, gyr


def _read_tum(path):
    d = np.loadtxt(path, comments="#", ndmin=2)
    q = d[:, 4:8] / np.linalg.norm(d[:, 4:8], axis=1, keepdims=True)
    x, y, z, w = q.T
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)
    return dict(t=d[:, 0], p=d[:, 1:4], R=R, name="mod-slam")


def load_local(orb_dir, inputs_dir, ref_dir=None, name=""):
    z = np.load(os.path.join(orb_dir, "trajectory.npz"), allow_pickle=False)
    t = z["timestamp_s"].astype(float)
    order = np.lexsort((t, z["segment"]))
    seg = Segment(
        name=name or os.path.basename(os.path.normpath(orb_dir)),
        t=t[order], p=z["position_m"].astype(float)[order], R_w_cam0=z["rotation"].astype(float)[order],
        map_id=z["segment"].astype(int)[order], t_imu=None, acc=None, gyr=None,
        calib=json.load(open(os.path.join(inputs_dir, "calibration.json"))), imu_offset_s=0.0, offset_source="none")
    seg.orb_dir, seg.order = orb_dir, order
    seg.t_imu, seg.acc, seg.gyr = _read_imu(os.path.join(inputs_dir, "imu.csv"))
    ft = os.path.join(inputs_dir, "frame_timestamps.csv")
    if os.path.exists(ft):
        seg.frame_t = np.loadtxt(ft, delimiter=",", skiprows=1, ndmin=2)[:, 1]
    # time offset: the run's own measurement first, the calibration prior last
    for fn in ("segments_manifest.json", "report.json"):
        p = os.path.join(orb_dir, fn)
        if os.path.exists(p):
            v = json.load(open(p)).get("measured_imu_time_offset_s")
            if v is not None:
                seg.imu_offset_s, seg.offset_source = float(v), fn
                break
    else:
        v = (seg.calib.get("imu") or {}).get("cam_imu_time_offset_s")
        if v is not None:
            seg.imu_offset_s, seg.offset_source = float(v), "calibration.json"
            seg.notes.append("no measured IMU time offset; used the calibration prior")
    if ref_dir and os.path.exists(os.path.join(ref_dir, "vio", "trajectory.txt")):
        seg.ref = _read_tum(os.path.join(ref_dir, "vio", "trajectory.txt"))
    return seg


def load_s3(bucket, segment, profile=None, workdir=None, with_ref=True):
    """segment = '<dataset>/<...>/<chunk>/seg_NNN' (the chunking key). Read-only."""
    import boto3
    from botocore.exceptions import ClientError
    s3 = (boto3.session.Session(profile_name=profile) if profile else boto3.session.Session()).client("s3")
    wd = workdir or tempfile.mkdtemp(prefix="orb_orient_")
    a, s = segment.rsplit("/", 1)

    def get(prefix, rel, dst_dir):
        dst = os.path.join(dst_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            s3.download_file(bucket, f"{prefix}/{rel}", dst)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    orb, inp, mod = (os.path.join(wd, x) for x in ("orbslam3", "inputs", "mod-slam"))
    for f in ORB_FILES:
        get(f"orbslam3/{a}/segments/{s}", f, orb)
    for f in INPUT_FILES:
        get(f"chunking/{segment}", f, inp)
    ref = None
    if with_ref:
        for pref in (f"mod-slam/{a}/segments/{s}", f"mod-slam/{segment}"):
            if get(pref, MOD_FILES[0], mod):
                ref = mod
                break
    return load_local(orb, inp, ref, name=segment)
