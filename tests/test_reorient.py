"""Synthetic test for reorient() + deliverable.write(): per-map re-orientation, no stitching.

Known head motion -> ORB-SLAM3-style deliverable (trajectory.npz + segments/*.txt + manifest,
three maps, each from (0,0,0) with a random heading, rotation = world <- raw cam0) -> reorient.
Checks: identical file names / keys / dtypes / rows / timestamps; every map still starts at
(0,0,0); the default trajectory.npz, rotated by the MCAP exporter's fixed R_x(-90), equals the A.2
camera sidecar; every map in the SAME axes (one world rotation fits all maps); gravity correct;
ORB-SLAM3's own folder untouched.

  python -m pytest tests/  (or: python tests/test_reorient.py)
"""
import hashlib
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from test_stitch import make_segment                               # noqa: E402
from src.core import Config                       # noqa: E402
from src.deliverable import write                 # noqa: E402
from src.io import load_local, read_npz           # noqa: E402
from src.reorient import reorient                 # noqa: E402


def _fake_orbslam3(seg, d):
    """Write seg the way ORB-SLAM3's stage does (run_clip.write_trajectory + make_deliverables_multi)."""
    os.makedirs(os.path.join(d, "segments"))
    files = []
    for k in np.unique(seg.map_id):
        s = seg.map_id == k
        fn = f"seg{k:02d}_map{k}_poses.txt"
        np.savetxt(os.path.join(d, "segments", fn), np.column_stack([seg.t[s], seg.p[s], seg.R_w_cam0[s].reshape(-1, 9)]),
                   fmt="%.9f", header="timestamp_s x y z_up r00..r22  (head=IMU origin, gravity-aligned, THIS SEGMENT'S OWN ORIGIN)")
        files.append(fn)
    np.savez_compressed(os.path.join(d, "trajectory.npz"), segment=seg.map_id.astype(np.int64), timestamp_s=seg.t,
                        position_m=seg.p, rotation=seg.R_w_cam0, segment_files=np.array(files),
                        frame_note=np.array("z is up, gravity-aligned; EACH SEGMENT HAS ITS OWN ORIGIN"))
    json.dump({"pose_frame": "imu", "measured_imu_time_offset_s": seg.imu_offset_s,
               "segments": [{"index": int(k), "map_id": int(k), "file": f"segments/{f}"} for k, f in enumerate(files)]},
              open(os.path.join(d, "segments_manifest.json"), "w"))
    inp = os.path.join(d, "..", "inputs")
    os.makedirs(inp)
    np.savetxt(os.path.join(inp, "imu.csv"), np.column_stack([seg.t_imu, seg.acc, seg.gyr]), delimiter=",",
               header="timestamp_s,ax,ay,az,gx,gy,gz", comments="", fmt="%.9f")
    json.dump(seg.calib, open(os.path.join(inp, "calibration.json"), "w"))
    return inp


def _digest(d):
    h = hashlib.sha256()
    for root, _, fs in sorted(os.walk(d)):
        for f in sorted(fs):
            h.update(open(os.path.join(root, f), "rb").read())
    return h.hexdigest()


def test_reorient_deliverable():
    seg0, tr = make_segment(cuts=((20.0, 0.13), (40.0, 1.5)))
    with tempfile.TemporaryDirectory() as tmp:
        orb = os.path.join(tmp, "orbslam3")
        inp = _fake_orbslam3(seg0, orb)
        before = _digest(orb)
        seg = load_local(orb, inp, name="synthetic")
        res = reorient(seg, Config(speed_p99_max=5.0))
        out = os.path.join(tmp, "out")
        write(res, seg, out)
        assert _digest(orb) == before, "ORB-SLAM3's folder was modified"

        a, b = read_npz(os.path.join(orb, "trajectory.npz")), read_npz(os.path.join(out, "trajectory.npz"))
        assert list(a) == list(b)
        for k in a:
            assert a[k].shape == b[k].shape and (a[k].dtype == b[k].dtype or k == "frame_note"), k
        for k in ("segment", "timestamp_s", "segment_files"):
            assert np.array_equal(a[k], b[k]), k
        assert sorted(os.listdir(os.path.join(orb, "segments"))) == sorted(os.listdir(os.path.join(out, "segments")))
        for k, fn in enumerate(b["segment_files"]):
            txt = np.loadtxt(os.path.join(out, "segments", str(fn)))
            s = b["segment"] == k
            assert np.allclose(txt[:, 0], b["timestamp_s"][s]) and np.allclose(txt[:, 1:4], b["position_m"][s], atol=1e-8)
            assert np.allclose(txt[:, 4:].reshape(-1, 3, 3), b["rotation"][s], atol=1e-8)
            assert np.allclose(b["position_m"][s][0], 0), "each map must keep its own origin"

        # default convention: trajectory.npz = world z-up <- IMU, which the MCAP exporter turns into
        # A.2 with its fixed R_x(-90). Emulate exactly that and compare with the A.2 camera sidecar.
        from src.reorient import R_ZUP_TO_A2
        c = read_npz(os.path.join(out, "trajectory_a2_camera.npz"))
        R_rect = tr["R_rect"]
        R_loader = np.einsum("ij,njk->nik", R_ZUP_TO_A2, b["rotation"])           # what load_orbslam3 publishes
        R_cam_from_loader = np.einsum("nij,kj->nik", R_loader, R_rect)
        d = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", R_cam_from_loader, c["rotation"]) - 1) / 2, -1, 1)))
        assert d.max() < 1e-4, f"loader-emulated pose != A.2 camera sidecar ({d.max():.2e} deg)"
        assert np.array_equal(c["timestamp_s"], b["timestamp_s"]) and list(c) == list(b)
        # the npz holds the IMU pose: gravity (specific force) must point to +z of its world
        f = np.stack([np.interp(b["timestamp_s"] + seg.imu_offset_s, seg.t_imu, seg.acc[:, i]) for i in range(3)], 1)
        up = np.einsum("nij,nj->ni", b["rotation"], f).mean(0)
        assert np.degrees(np.arccos(up[2] / np.linalg.norm(up))) < 0.5
        # all maps in ONE world (A.2 camera sidecar): a single rotation maps true world to output
        R_out = c["rotation"]
        R_true_h = np.einsum("nij,kj->nik", tr["R_wi"], tr["R_rect"])
        D0 = R_out[0] @ R_true_h[0].T
        err = [np.degrees(np.arccos(np.clip((np.trace(R_out[i] @ (D0 @ R_true_h[i]).T) - 1) / 2, -1, 1)))
               for i in range(len(R_out))]
        assert np.max(err) < 1.0, f"maps do not share one frame: max {np.max(err):.2f} deg"
        up = D0 @ np.array([0, 0, 1.0])
        assert np.degrees(np.arccos(-up[1])) < 0.5, "true up is not -Y"
        assert R_out[0][1, 1] > 0                                   # camera +Y (image down) points down-ish
        # world +Z = horizontal facing at the start: the camera's forward axis has no X component there
        f = R_out[: 30, :, 2].mean(0)
        assert abs(np.degrees(np.arctan2(f[0], f[2]))) < 2.0
        rep = json.load(open(os.path.join(out, "orientation_report.json")))
        assert len(rep["maps"]) == 3 and len(rep["bridges"]) == 2
        return float(np.max(err))


def test_guards():
    """Reviewer findings E5-E8: each guard must fire on the failure it exists for."""
    from src.geometry import exp_so3
    from src.simulate import level_like_exporter
    cfg = Config(speed_p99_max=5.0)
    # E6: exporter switched to the RECTIFIED extrinsic -> identified exactly, not missed
    seg, _ = make_segment()
    R_rect = np.array(seg.calib["rectified_extrinsics"]["T_cam0rect_imu"])[:3, :3]
    R_raw = np.array(seg.calib["imu"]["T_cam0_imu"])[:3, :3]
    seg.R_w_cam0 = np.einsum("nij,jk,lk->nil", seg.R_w_cam0, R_raw, R_rect)    # world <- rect cam0
    level_like_exporter(seg, R_rect)
    r = reorient(seg, cfg)
    assert all(m["export_convention"] == "rectified_cam0" for m in r["maps"])
    assert any("convention changed upstream" in a for a in r["alarms"])
    # E6: an unknown convention (rotation off by 0.3 deg) -> not trusted, alarm
    seg, _ = make_segment()
    seg.R_w_cam0 = np.einsum("nij,jk->nik", seg.R_w_cam0, exp_so3([0.005, 0, 0]))
    r = reorient(seg, cfg)
    assert all(m["export_convention"] == "unrecognised" and not m["orientation_trusted"] for m in r["maps"])
    assert any("not recognised" in a for a in r["alarms"])
    # N1: not exact but a clear margin -> identified approximately (trusted, alarm), not unrecognised
    seg, _ = make_segment()
    seg.R_w_cam0 = np.einsum("ij,njk->nik", exp_so3([np.radians(0.01), 0, 0]), seg.R_w_cam0)
    r = reorient(seg, cfg)
    assert all(m["export_convention"] == "raw_cam0_approx" and m["orientation_trusted"] for m in r["maps"])
    assert any("only approximately" in a for a in r["alarms"])
    # N1: no clear margin (raw 0.02 deg vs rectified 0.03 deg) -> unrecognised, never a bare argmin
    seg, _ = make_segment()
    seg.calib = json.loads(json.dumps(seg.calib))
    T = np.array(seg.calib["imu"]["T_cam0_imu"])
    T[:3, :3] = exp_so3([0, np.radians(0.02), 0]) @ T[:3, :3]
    seg.calib["rectified_extrinsics"]["T_cam0rect_imu"] = T.tolist()
    seg.R_w_cam0 = np.einsum("ij,njk->nik", exp_so3([np.radians(0.02), 0, 0]), seg.R_w_cam0)
    r = reorient(seg, cfg)
    assert all(m["export_convention"] == "unrecognised" for m in r["maps"])
    # E5: a map whose tilt drifts (2 deg over its length) -> windowed gravity flags it,
    # although its whole-map mean is levelled to 0
    seg, _ = make_segment()
    s = seg.map_id == 0
    ang = np.radians(4.0) * (seg.t[s] - seg.t[s][0]) / (seg.t[s][-1] - seg.t[s][0]) - np.radians(2.0)
    seg.R_w_cam0[s] = np.stack([exp_so3([a, 0, 0]) @ R for a, R in zip(ang, seg.R_w_cam0[s])])
    level_like_exporter(seg)
    r = reorient(seg, Config(speed_p99_max=5.0, grav_p90_max_deg=1.0))
    m0 = [m for m in r["maps"] if m["map_id"] == 0][0]
    assert m0["gravity_window_p90_deg"] > 1.0 and not m0["orientation_trusted"], m0
    # E8: every other ORB-SLAM3 file is carried over, and the output is marked as already A.2
    seg0, _ = make_segment()
    with tempfile.TemporaryDirectory() as tmp:
        orb = os.path.join(tmp, "orbslam3")
        inp = _fake_orbslam3(seg0, orb)
        json.dump({"status": "ok"}, open(os.path.join(orb, "status.json"), "w"))
        seg = load_local(orb, inp)
        out = os.path.join(tmp, "out")
        write(reorient(seg, cfg), seg, out)
        assert json.load(open(os.path.join(out, "status.json"))) == {"status": "ok"}
        man = json.load(open(os.path.join(out, "segments_manifest.json")))
        assert man["pose_frame"] == "imu" and man["axis_convention"] == "zup_then_rx_minus90_to_opeth_a2"
        assert str(read_npz(os.path.join(out, "trajectory.npz"))["frame_note"]).startswith("ORB_ORIENTATION imu_zup")
        assert str(read_npz(os.path.join(out, "trajectory_a2_camera.npz"))["frame_note"]).startswith("OPETH_A2")
        out2 = os.path.join(tmp, "out2")                         # the other convention on request
        write(reorient(seg, cfg), seg, out2, "a2-camera")
        man2 = json.load(open(os.path.join(out2, "segments_manifest.json")))
        assert man2["axis_convention"] == "opeth_a2" and os.path.exists(os.path.join(out2, "trajectory_imu_zup.npz"))


if __name__ == "__main__":
    print(f"ok: max orientation error across all maps {test_reorient_deliverable():.3f} deg")
    test_guards()
    print("ok: guards (rectified convention, unknown convention, approximate / ambiguous convention, "
          "tilt drift, file carry-over, markers)")
