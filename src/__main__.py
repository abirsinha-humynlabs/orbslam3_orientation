"""Command line.

  # per-map re-orientation, SAME deliverable files as ORB-SLAM3 (no stitching)
  python -m src reorient --orbslam3 ORB_DIR --inputs CHUNK_DIR --out OUT [--render OUT/check.mp4 --video V]
  python -m src reorient --bucket B --profile P --segment <chunking key> --out OUT

  # local folders
  python -m src process --orbslam3 ORB_DIR --inputs CHUNK_DIR [--mod-slam MOD_DIR] --out OUT
  # straight from S3 (read-only)
  python -m src process --bucket prod-egc-stereo-v2-data --profile prod \
      --segment bitrobot/<site>/<date>/<worker>/<session>/<chunk>/seg_NNN --out OUT
  # also render a check video (needs opencv; --video can be a local left_rectified.mp4)
  python -m src process ... --render OUT/check.mp4 [--video left_rectified.mp4]

Exit code: 0 ok, 1 no usable map, 2 ok but with an alarm (frame-convention change, poor stitch,
gravity self-check failure).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from . import Config, load_local, load_s3, process
from .output import save


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("process", "stitch the maps into one continuous trajectory"),
                      ("reorient", "re-orient every map to Opeth's axes, same files/structure as ORB-SLAM3")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--orbslam3", help="local orbslam3 output dir (trajectory.npz, segments_manifest.json)")
        p.add_argument("--inputs", help="local chunking dir (imu.csv, calibration.json, frame_timestamps.csv)")
        p.add_argument("--bucket", help="S3 bucket (read-only)")
        p.add_argument("--segment", help="segment key under chunking/")
        p.add_argument("--profile", default=None, help="AWS profile")
        p.add_argument("--out", required=True)
        p.add_argument("--render", help="also write a check video here")
        p.add_argument("--video", help="left_rectified.mp4 for --render (S3 mode downloads it if omitted)")
        if name == "reorient":
            p.add_argument("--convention", choices=("mcap", "a2-camera"), default="mcap",
                           help="trajectory.npz convention: mcap (default; world z-up <- IMU, what the MCAP "
                                "exporter expects) or a2-camera (Opeth A.2 <- rectified camera); the other is "
                                "written as a sidecar npz")
        if name == "process":
            p.add_argument("--mod-slam", help="optional local mod-slam dir (vio/trajectory.txt) to bridge long gaps")
            p.add_argument("--no-reference", action="store_true", help="do not use mod-slam for long gaps")
        for k, v in Config().__dict__.items():
            p.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    a = ap.parse_args(argv)

    cfg = Config(**{k: getattr(a, k) for k in Config().__dict__})
    use_ref = a.cmd == "process" and not a.no_reference
    if a.bucket:
        wd = tempfile.mkdtemp(prefix="orb_orient_")
        seg = load_s3(a.bucket, a.segment, a.profile, wd, with_ref=use_ref)
    else:
        if not (a.orbslam3 and a.inputs):
            ap.error("give --orbslam3 and --inputs, or --bucket and --segment")
        seg = load_local(a.orbslam3, a.inputs, a.mod_slam if use_ref else None,
                         name=a.segment or os.path.basename(os.path.normpath(a.orbslam3)))
    if a.cmd == "reorient":
        return _reorient(a, ap, seg, cfg, wd if a.bucket else None)
    res = process(seg, cfg)
    paths = save(res, a.out)
    summary = dict(segment=res["segment"], ok=res["ok"], n_maps=len(res["maps"]),
                   kept=[m["map_id"] for m in res["maps"] if m["kept"]], gaps=len(res["gaps"]),
                   gravity_check_deg=res.get("gravity_check_deg"), alarms=res["alarms"], files=paths)
    if res["ok"] and a.render:
        from .render import render
        video = a.video
        if not video and a.bucket:
            import boto3
            video = os.path.join(wd, "left_rectified.mp4")
            s3 = (boto3.session.Session(profile_name=a.profile) if a.profile else boto3.session.Session()).client("s3")
            s3.download_file(a.bucket, f"chunking/{a.segment}/left_rectified.mp4", video)
        if not video:
            ap.error("--render needs --video in local mode")
        render(res, seg, video, a.render)
        summary["render"] = a.render
    print(json.dumps(summary, indent=2))
    if not res["ok"]:
        return 1
    return 2 if res["alarms"] else 0


def _video(a, ap, wd):
    if a.video:
        return a.video
    if not a.bucket:
        ap.error("--render needs --video in local mode")
    import boto3
    video = os.path.join(wd, "left_rectified.mp4")
    s3 = (boto3.session.Session(profile_name=a.profile) if a.profile else boto3.session.Session()).client("s3")
    s3.download_file(a.bucket, f"chunking/{a.segment}/left_rectified.mp4", video)
    return video


def _reorient(a, ap, seg, cfg, wd):
    from .deliverable import write
    from .reorient import reorient
    res = reorient(seg, cfg)
    paths = write(res, seg, a.out, a.convention)
    summary = dict(segment=res["name"], n_maps=len(res["maps"]), heading_anchor_map=res["anchor_map"],
                   healthy_maps=res["healthy_maps"], flagged_maps=res["unhealthy_maps"],
                   bridges=[(b["from_map"], b["to_map"], b["bridge_s"], b["quality"]) for b in res["bridges"]],
                   gravity_window_p90_deg={m["map_id"]: m["gravity_window_p90_deg"] for m in res["maps"]},
                   alarms=res["alarms"], files=sorted(paths))
    if a.render:
        from .render import render_maps
        render_maps(res, seg, _video(a, ap, wd), a.render)
        summary["render"] = a.render
    print(json.dumps(summary, indent=2))
    return 2 if res["alarms"] else 0


if __name__ == "__main__":
    sys.exit(main())
