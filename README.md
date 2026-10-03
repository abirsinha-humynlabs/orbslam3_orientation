# orbslam3_orientation

Post-processes ORB-SLAM3 output into the axis convention Opeth requires (**+X right, +Y down,
+Z forward**), **without changing ORB-SLAM3**. It reads what the orbslam3 stage already
publishes and the segment's IMU and calibration.

Two modes:

| Mode | What it writes | Use |
| --- | --- | --- |
| **`reorient`** (the delivered one) | the **same files, names, keys, rows and map structure** as ORB-SLAM3 (`trajectory.npz`, `segments/segNN_mapK_poses.txt`, …), only re-oriented | downstream reads it instead of ORB-SLAM3's own output |
| `process` | one stitched, continuous trajectory (`poses.csv/npz`) | experimental, not used downstream |

## Why it's needed

ORB-SLAM3's published trajectory (`orbslam3/<seg>/segments/seg_NNN/trajectory.npz`) is:

| | What it is | What it's labelled |
| --- | --- | --- |
| `rotation` | world ← **raw** left camera (cam0) | `pose_frame: "imu"` |
| `position_m` | IMU origin | IMU |
| world | gravity-aligned, +z up, x/y heading arbitrary | – |
| maps | each atlas map (`segment`) restarts at (0,0,0) **with a new heading** | – |

This was verified on 201 runs from all 51 dataset/site/date directories, and confirmed in the
stage code (`make_deliverables_multi.py`, `load_segment`). Used as-is:

- the head orientation is about 100° off if treated as the IMU frame;
- the trajectory jumps back to the origin and turns by a median of 84° at every map switch;
- "forward" is whatever heading ORB-SLAM3 happened to start with.

## `reorient`: per-map re-orientation, same deliverable

ORB-SLAM3 often splits a video into several atlas maps (a 10-minute video may have 3). They stay
separate; **no stitching, no pose added or removed, positions restart at (0,0,0) in every map**,
exactly as delivered. Only the axes change, and **every map gets the same axes**:

| | ORB-SLAM3 as delivered | After `reorient` |
| --- | --- | --- |
| world axes | +z up; x/y heading arbitrary and **different in every map** | **+X right, +Y down (gravity), +Z forward**, the **same heading in every map** |
| world +Z | – | the operator's horizontal facing over the first second of the heading-anchor map (the first healthy map; usually map 0) |
| `rotation` | world ← raw left camera (labelled "imu") | world ← **rectified left camera** (the camera of `left_rectified.mp4`: +X image right, +Y image down, +Z optical axis) |
| `position_m` | IMU origin, per-map origin | rectified left camera centre, per-map origin (0,0,0 at the map's first pose) |
| `timestamp_s`, `segment`, `segment_files` | – | unchanged, bit for bit |

### How a common heading is given to every map

1. Per map:
   - `R_world_imu = R_npz · T_cam0_imu[:3,:3]`.
   - **Exact export-convention guard.** With the extrinsic the exporter used, the levelling
     residual is exactly 0; that identifies raw vs rectified, and anything else is flagged.
   - Re-level to the measured gravity.
   - **Health:**
     - speed, at least 3 s long;
     - 3-axis rate residual ≤ 0.25 of the gyro rate and correlation ≥ 0.95 (trajectory vs bias-corrected gyro);
     - windowed gravity residual (10 s) p90 ≤ 2°.
   - **Maps that fail are kept and flagged, not removed.** See `docs/how_it_works.md` §2 and §11.
2. **Heading anchor:** the first healthy map. Its first second defines world +Z.
3. **Hand-off across each map break, by the gyroscope:** integrate the bias-corrected gyro from
   0.5 s inside the previous reliable *healthy* map (poses right at a tracking loss are often off) to the
   first 1.5 s of the next map. The heading difference (circular mean) is applied to the whole
   next map. Maps before the anchor are bridged backwards in time.
   - **Gyro bias:** median of (gyro − trajectory body rate) over 2 s windows of every map whose
     orientation tracks the gyro. These IMUs have biases of about 0.045 rad/s (2.6°/s), so this
     is essential.
   - **Consistency check:** both maps are levelled independently by gravity, so the gyro's
     predicted tilt must match the next map's. If it's off by more than 5°, one of the two maps
     is wrong near the break. The next reliable map is then tried; if none passes, the
     hand-off is flagged `poor` and `heading_shared_reliably: false`.
4. Express all maps in the A.2 world; the head frame is the rectified left camera.

### Output (`--out`): mirrors ORB-SLAM3's folder

| File | Content |
| --- | --- |
| `trajectory.npz` | same keys (`segment, timestamp_s, position_m, rotation, segment_files, frame_note`), dtypes, row count and order; `rotation` and `position_m` re-oriented, `frame_note` updated |
| `segments/segNN_mapK_poses.txt` | same names, `timestamp_s x y z r00..r22` (row-major), `%.9f`, header updated |
| `segments_manifest.json` | copied; extents recomputed in the new axes; explicit markers `rotation_frame`/`position_frame` = `cam0_rectified`, `axis_convention` = `opeth_a2` (never re-rotate); a `post_process` block added |
| `report.json`, `status.json`, … | every other file of ORB-SLAM3's folder copied unchanged |
| `orientation_report.json` | **new sidecar.** Conventions; per map: `healthy` + `health_issues`, `export_convention`, rate and windowed-gravity metrics, `orientation_trusted`, `heading_anchor`, `heading_source`, `heading_shared_reliably`, `yaw_applied_deg`; per hand-off: gap, bridge length, rotation during the bridge, yaw, spread, tilt residual, `quality` (`good` / `fair` / `approximate` / `poor`); segment gyro bias; alarms |

ORB-SLAM3's own folder is never modified (the writer refuses `--out` equal to the source).

### Validation

**Synthetic** (`tests/test_reorient.py`): three maps with random headings. One rotation fits all
maps to within **0.26°**; +Y is gravity; structure is identical; the source folder is untouched.

**Real data, cut test** (`tools/validate_reorient.py cut`): 70 healthy single-map runs, each cut
into two maps with a random heading and origin. The error is the heading-transfer error between
the two maps.

| Gap | Hand-offs graded reliable: median / p95 / worst | All hand-offs: median / p95 |
| --- | --- | --- |
| 0.13 s | 0.37° / 0.86° / 1.23° | 0.37° / 0.96° |
| 1 s | 0.37° / 1.08° / 1.53° | 0.38° / 1.07° |
| 5 s | 0.44° / 1.70° / 2.53° | 0.48° / 1.95° |
| 30 s | (always graded `poor`) | 0.88° / 5.2° |

**Real map breaks vs mod-slam** (`tools/validate_reorient.py real`): mod-slam is one continuous
map, independent of ORB-SLAM3. The heading offset to mod-slam is compared 2 s before vs 2 s after
each hand-off between two orientation-trusted maps:
- before the review fixes, on 10 runs: graded reliable, median **3.0°**, p90 7.9° (n=8); flagged,
  median 15.5°, max 74° (n=9);
- after them, only 3 hand-offs remain between two trusted maps: 0.25° (PLN-024, 3.8 s gap),
  7.7° (27.6 s gap) and 8.3° (84 s gap, flagged).

Real breaks happen during violent head motion: 150–2000° of rotation inside the gap. So
long-gap heading is uncertain at the several-degree level, and the grade says so. Measured
directly, gyro vs mod-slam across real breaks: under 1 s gaps ~0.15°; 1–10 s gaps ~2.6° median.

**Axes against the footage.**
- Optical flow in `left_rectified.mp4` matches the gyro mapped into the rectified camera (both
  image axes, positive slopes).
- The accelerometer's gravity in that camera frame matches the video posture: looking ahead
  gives image-up = up. Looking at the lap while seated, the camera points past vertical and
  image-down legitimately has an upward component.

**Batch results** (all 39 cached multi-map runs, 113 maps, after the review fixes of 2026-10-03):
- All 39 outputs have the same structure as ORB-SLAM3's: keys, shapes and timestamps.
- Export convention identified exactly as `raw_cam0` on all 113 maps.
- 16 maps are healthy; 26 maps (including the heading anchors) have their heading reliably shared.
  Before the stricter orientation tests and the healthy-only rule for heading sources, this was 48.
- Hand-off grades: 3 good, 7 fair, 9 approximate, 55 poor.
- 24 runs raise an alarm, mostly "no healthy map". The windowed gravity residual (p90 per map)
  has median 2.6° and max 33°; most maps are ORB-SLAM3 tracking failures.

### Usage

```bash
pip install -r requirements.txt

# local folders (ORB-SLAM3's segment folder + chunking inputs)
python -m orbslam3_orientation reorient \
    --orbslam3 <orbslam3 seg dir> --inputs <chunking seg dir> --out <new folder> \
    [--render check.mp4 --video left_rectified.mp4]

# straight from S3 (read-only: only downloads; output is local)
python -m orbslam3_orientation reorient --bucket prod-egc-stereo-v2-data --profile prod \
    --segment <dataset>/.../<chunk>/seg_NNN --out <new folder> [--render check.mp4]

python tests/test_reorient.py
python tools/validate_reorient.py cut  <segdir> ...
python tools/validate_reorient.py real <segdir> ...
```

Exit code: 0 ok, 2 written but with an alarm (e.g. no map with a trustworthy orientation).
Every threshold is a flag (`--help`).

**Check video (`--render`).**
- Left: the video, with the world triad drawn 1 m in front of the camera (X red, Y green,
  Z blue) and the horizon. Across a map switch the triad must keep pointing the same way in
  the scene, and green must point to gravity-down.
- Right: each map from its own origin in the shared axes (top view, X right, Z up on screen);
  ORB-SLAM3's raw maps; a timeline with the hand-off grades. Grey = flagged map.

### Mapping to Opeth's MCAP topics

| Topic | From this output |
| --- | --- |
| `/ego/vio/pose` | parent `ego_vio_world`, child `ego_head` (rectified left camera); one pose per row; a new map = a new origin (start a new `ego_vio_world_<map>` frame, or reset; the axes are shared) |
| `/ego/vio/system_info` | `axis_convention`, world/head frame names, per-map health and hand-off grades from `orientation_report.json` |
| `/tf_static` | `ego_head` → IMU from `calibration.json` (`rectified_extrinsics.T_cam0rect_imu`) |

### Limitations

- **The heading across a break is only as good as the gyro over the gap.** Under 1.5 s it's
  sub-degree. Over long gaps with a lot of head rotation it's several degrees, and graded.
- **A flagged (diverged) map** is re-oriented like the others, but its own poses are
  ORB-SLAM3's failure. Use the sidecar's `healthy` / `heading_shared_reliably` to decide.
- **Agreement isn't truth.** A wrong calibration fools ORB-SLAM3, the IMU and the checks alike.

---

## `process` (experimental): one stitched trajectory

### What it produces

Opeth's *Human Egocentric Videos* spec, Appendix A.2 (operator perspective):

| Axis | Convention |
| --- | --- |
| +X | right |
| +Y | down (gravity) |
| +Z | forward |

- **World frame `ego_vio_world`, one per video segment.**
  - Origin: the head position at the first pose.
  - +Y: gravity.
  - +Z: the operator's horizontal facing direction over the first second.
  - +X = +Y × +Z. Right-handed.
- **Head frame `ego_head`:** the rectified left camera (`top-left-camera`). Its axes are
  already +X image-right, +Y image-down, +Z optical axis.
- **IMU frame:** also given (`q_world_imu`), for consumers who want the body frame.

### How it works

1. **Per map:**
   - `R_world_imu = R_npz · T_cam0_imu[:3,:3]`.
   - **Frame guard:** gravity must read within 0.5° of vertical. If it doesn't, but the
     rectified extrinsic fits, ORB-SLAM3's convention has changed upstream, so the map is
     dropped and an alarm raised.
   - **Re-level** to the measured gravity, so the result doesn't depend on ORB-SLAM3 having done it.
   - **Health check:** speed p99 ≤ 3 m/s and max ≤ 15 m/s at 10 Hz, gyro correlation ≥ 0.9,
     and at least 3 s long. Maps that fail (diverged) are dropped.
2. **Stitch** the kept maps in time order:
   - **Heading:** integrate the gyroscope across the gap, after removing its bias, which is
     estimated over the 10 s of trajectory just before the gap. The heading difference to the
     new map's first pose is applied to the whole new map.
   - **Position:**
     - gaps ≤ 2 s: constant velocity;
     - longer gaps: a reference trajectory (mod-slam), when one covers the gap;
     - otherwise: held at the last position and flagged.
   - **No poses are emitted inside gaps.**
3. **Build the operator world** and express the head and IMU poses in it.
4. **Self-check:** the mean specific force must point to −Y in the output. Otherwise an alarm
   is raised.

### Validation

**Synthetic** (`tests/test_stitch.py`): a known head motion with exact IMU readings, cut into
three maps with random headings (up to 162° apart). Max orientation error is **0.26°** and max
position error **4.4 cm**.

**Real data, cut test** (`tools/validate_stitch.py cut`): 72 healthy single-map runs, each cut
once per gap length, with the second part's heading and origin scrambled, run through
`process()`:

| Gap | Orientation error after the gap (median / p95) | Position error (median / p95) |
| --- | --- | --- |
| 0.13 s | 0.16° / 0.53° | 0.03 m / 0.11 m |
| 1 s | 0.19° / 0.52° | 0.17 m / 0.72 m |
| 5 s | 0.40° / 1.69° | 0.30 m / 3.8 m (no reference; held) |
| 30 s | 1.27° / 6.48° | 0.41 m / 9.1 m (no reference; held) |

Real map breaks, from 74 breaks in 201 runs: 34 are under 0.5 s, 4 are 0.5–2 s, 26 are
2–10 s, and 10 are longer.

**Real multi-map runs** (`tools/validate_stitch.py real`): after stitching, each map's heading
matches the continuous mod-slam trajectory of the same clip to within 0.05° (PLN-024, 3.8 s
gap) and 0.28° (PIP-305, 1.4 s gap).

Most multi-map runs are tracking failures: of 113 maps in 39 multi-map runs, 96 fail the
health check (median p99 speed 23.5 m/s). Those are dropped and listed.

### Usage

```bash
pip install -r requirements.txt

# local folders
python -m orbslam3_orientation process \
    --orbslam3 <orbslam3 seg dir> --inputs <chunking seg dir> [--mod-slam <mod-slam seg dir>] \
    --out out/<seg> [--render out/<seg>.mp4 --video <left_rectified.mp4>]

# straight from S3 (read-only: only downloads)
python -m orbslam3_orientation process --bucket prod-egc-stereo-v2-data --profile prod \
    --segment bitrobot/<site>/<date>/<worker>/<session>/<chunk>/seg_NNN --out out/<seg> [--render out/<seg>.mp4]

python tests/test_stitch.py
python tools/validate_stitch.py cut  <segdir> ...
python tools/validate_stitch.py real <segdir> ...
```

Every threshold is a flag; run `--help`. Exit code: 0 ok, 1 no usable map, 2 ok but with an alarm.

The required input files per segment:
- `orbslam3/`: `trajectory.npz`, plus `segments_manifest.json` or `report.json`
- `chunking/`: `imu.csv`, `calibration.json`, `frame_timestamps.csv`
- optional mod-slam: `vio/trajectory.txt`

### Output files (`--out`)

| File | Content |
| --- | --- |
| `poses.csv` | `t_camera_s, t_imu_s, map_id, x_m, y_m, z_m, head_qx..qw, imu_qx..qw` (quaternions x, y, z, w; world ← frame) |
| `poses.npz` | the same, plus the rotation matrices |
| `orientation.json` | conventions, per-map checks (kept/dropped and why), per-gap stitching (gap length, yaw applied, gyro bias, position method, quality), alarms, gravity self-check |

### Mapping to Opeth's MCAP topics

| Topic | From this output |
| --- | --- |
| `/ego/vio/pose` (foxglove.FrameTransforms) | parent `ego_vio_world`, child `ego_head`; translation = `x,y,z`, rotation = `head_q*`; stamped with `t_imu` if the MCAP runs on the IMU clock |
| `/ego/vio/system_info` (RobotInfo metadata) | `world_frame=ego_vio_world`, `head_frame=ego_head`, `axis_convention` and `world_definition` from `orientation.json`, plus the map and gap summary |
| `/tf_static` | `ego_head` → IMU from `calibration.json` (`rectified_extrinsics.T_cam0rect_imu`) |
| `/ego/vio/relative_pose` | consecutive-pose deltas, computed only within a run of poses (never across a gap) |

### Limitations

- **Heading across long gaps** drifts with the gyro: p95 1.7° at 5 s, 6.5° at 30 s. Each gap
  is graded `good` / `fair` / `approximate` / `poor` in `orientation.json`.
- **Position across gaps over 2 s** needs a reference trajectory. Without one it's held and
  flagged `unknown`.
- **Agreement isn't truth.** A wrong calibration fools ORB-SLAM3, the IMU checks and the
  reference alike.
- **The world's +Z** is defined from the first second's facing direction. If Opeth expects a
  different world definition, only step 3 changes.
