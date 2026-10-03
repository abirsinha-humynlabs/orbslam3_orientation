# ORB-SLAM3 orientation post-process: how it works

**Code:** `s3://stage-humyn-egocentric-stereo-data/labelling_results/orb_orientation/code/`
(package `orbslam3_orientation`, version 1.1.0)

**Check videos:** `s3://stage-humyn-egocentric-stereo-data/labelling_results/orb_orientation/<clip>.mp4`,
each with its re-oriented output folder `<clip>/`.

## 1. Purpose

ORB-SLAM3's trajectory doesn't match the axis convention Opeth requires (Human Egocentric Videos
spec, Appendix A.2, operator perspective):

| Axis | Opeth A.2 |
| --- | --- |
| +X | right |
| +Y | down (gravity) |
| +Z | forward |

This tool **re-orients ORB-SLAM3's output into those axes without changing ORB-SLAM3**. Its
output has the same file names and structure as ORB-SLAM3's, so downstream reads it **in place
of** ORB-SLAM3's own output.

## 2. What ORB-SLAM3 delivers (verified on 201 runs from 51 directories, and in the exporter code)

Per segment: `orbslam3/<...>/segments/seg_NNN/`

| File | Content |
| --- | --- |
| `trajectory.npz` | `segment` (map index per row), `timestamp_s` (camera clock), `position_m`, `rotation` (N×3×3), `segment_files`, `frame_note` |
| `segments/segNN_mapK_poses.txt` | the same rows per map: `timestamp_s x y z r00..r22` |
| `segments_manifest.json`, `report.json` | map list, time offset, health stats |

What the arrays actually mean:

- **`rotation` = world ← RAW left camera (cam0)**, although the manifest says `pose_frame: "imu"`.
  The exporter computes `R_wc = R_wb · T_cam0_imuᵀ` with the raw (not rectified) extrinsic.
- **`position_m` = IMU origin.**
- **World:** gravity-aligned, +z up, but **the x/y heading is arbitrary and different in every map**.
- **Multiple maps:** when tracking is lost, ORB-SLAM3 starts a new atlas map, so a 10-minute
  video can have several. Every map restarts at (0,0,0) with a new heading (median yaw jump at a
  switch: 84°). 39 of 201 runs have several maps (74 switches).

## 3. What the tool writes (`reorient` mode, the delivered one)

A folder that mirrors ORB-SLAM3's:

| File | Change |
| --- | --- |
| `trajectory.npz` | **Same keys, dtypes, row count and row order.** `timestamp_s`, `segment` and `segment_files` are identical. `rotation` and `position_m` are re-oriented. `frame_note` starts with the marker `OPETH_A2 cam0_rectified`. |
| `segments/segNN_mapK_poses.txt` | Same names and columns, `%.9f`. New header. |
| `segments_manifest.json` | Copied. Per-map extents recomputed in the new axes. **Explicit markers:** `pose_frame` / `rotation_frame` / `position_frame` = `cam0_rectified`, `world_frame` = `axis_convention` = `opeth_a2`, `gravity_axis` = `+y_down`. A `post_process` block is added. A reader must dispatch on these and **never** apply the OKVIS → A.2 rotation again *(E8)*. |
| `report.json`, `status.json`, … | **Every other file** in ORB-SLAM3's folder is copied unchanged, so readers find everything they expect *(E8)*. |
| `orientation_report.json` | **New sidecar:** conventions, per-map health, heading hand-offs, alarms (section 6). |

The meaning after re-orientation:

| | Meaning |
| --- | --- |
| world axes | **+X right, +Y down (gravity), +Z forward; the same heading in EVERY map** |
| world +Z | the operator's horizontal facing direction during the first second of the *heading-anchor map* (the first healthy map; usually map 0) |
| `rotation` | world ← **rectified left camera**, the camera of `left_rectified.mp4`: +X image right, +Y image down, +Z optical axis. Its columns are the camera axes in world coordinates. |
| `position_m` | the rectified left camera's centre. **Each map keeps its own origin: (0,0,0) at its first pose.** |
| `timestamp_s` | unchanged (camera clock). IMU clock = `timestamp_s + measured_imu_time_offset_s`. |

**No stitching.** No pose is added, removed or moved between maps. Maps stay separate, as
ORB-SLAM3 delivered them. Only the axes change, and all maps now share them.

## 4. Pipeline

```
trajectory.npz ─┐
imu.csv ────────┼─► 1. load ─► 2. per-map checks + re-level ─► 3. heading anchor
calibration ────┘                                                 │
                                                                  ▼
 7. write deliverable ◄─ 6. A.2 world + head frame ◄─ 5. hand-offs ◄─ 4. gyro bias
```

### Step 1: load (`io.py`)
- Read `trajectory.npz`, keeping its original row order for the write-back.
- Read the IMU (`imu.csv`, ~200 Hz), `calibration.json` and `frame_timestamps.csv`.
- **IMU time offset:** `segments_manifest.json` → `report.json` → calibration prior, in that
  order of preference.
- Inputs can be local folders or S3 (read-only).

### Step 2: per-map checks and re-levelling (`core._check_map`, `reorient._orientation_metrics`)
For each map:

- **Export-convention guard (exact).** The exporter levels each map with `mean(R_wb · acc)`, so
  with the extrinsic it really used, the residual tilt is **exactly 0** (below 1e-6° on 133 maps).
  The other extrinsic gives at least 0.0045°. Using a 1e-4° threshold:

  | Result | Meaning | Action |
  | --- | --- | --- |
  | `raw_cam0` | today's exporter | `R_world_imu = R_npz · T_cam0_imu[:3,:3]` |
  | `rectified_cam0` | the exporter switched extrinsic | identified exactly, used, alarm raised |
  | `unrecognised` | anything else | map not trusted, alarm raised |

  A tolerance such as 0.5° can't do this: 49 of 132 maps have a rectified-extrinsic residual
  below 0.5°. *(Reviewer finding E6.)*
- **Re-level:** the smallest rotation that puts the measured gravity exactly on +z.
- **Position health:** at least 3 s long; speed at 10 Hz p99 ≤ 3 m/s and max ≤ 15 m/s.
- **Orientation tests**, independent of position:

  | Test | Limit | Calibrated on 145 healthy / 102 diverged maps | Finding |
  | --- | --- | --- | --- |
  | 3-axis rate residual: RMS of (trajectory body rate − (gyro − bias)), relative to the gyro's RMS rate | ≤ 0.25 | healthy median 0.06, p95 0.20; diverged median 0.36 | E7 |
  | correlation of the 3 rate components | ≥ 0.95 | healthy p5 0.98 | E7 |
  | windowed gravity residual: mean specific force per 10 s window vs the vertical, p90 over windows | ≤ 2° | healthy p99 1.7; diverged median 2.1 | E5 |

  The bias for the rate test is the median over the position-healthy maps. The old whole-map
  gravity check was forced to 0 by the levelling, and the old magnitude-only rate correlation
  ignored axis errors.
- A map is **healthy** only if it passes everything. Failing maps are **kept and flagged**.

### Step 3: heading anchor
- The first map in time that is healthy **and** has a trusted orientation. Fallback: the first
  trusted map, then the first map (with an alarm).
- Its heading is the reference: yaw 0.

### Step 4: gyro bias (`reorient._segment_bias`)
- These IMUs have gyro biases of about **0.045 rad/s (2.6°/s)**, so integrating the raw gyro
  is useless.
- Bias = **median over 2 s windows** of (gyro − rotation rate of the trajectory), pooled over
  every orientation-trusted map in the segment. Body rates don't depend on a map's heading, so
  maps can be pooled as they are.
- On 28 real map breaks checked against mod-slam, this beat a bias taken right next to the gap
  (1–10 s gaps: 2.6° vs 4.4° median heading error).

### Step 5: heading hand-off across each map break (`reorient.reorient → bridge/place`)
For every other map, in time order away from the anchor:

1. **Source pose:** 0.5 s inside the nearest reliable map (poses right at a tracking loss are
   often wrong).
2. **Predict:** integrate the bias-corrected gyro from that pose to each pose in the first 1.5 s
   of the new map. Maps before the anchor are bridged backwards.
3. **Yaw:** compare the prediction with the new map's own orientation. The yaw difference
   (circular mean) is applied to the **whole** new map, so it gets the shared heading.
4. **Consistency check:** both maps were levelled independently by gravity, so the predicted
   tilt must match the new map's tilt. If the residual tilt or the yaw spread is over 5°, one of
   the two maps is wrong near the break. The next reliable map is then tried; if none passes,
   the hand-off is graded `poor` and the map gets `heading_shared_reliably: false`.
5. **Grade:**

   | Grade | Bridge length | Head rotation during the bridge |
   | --- | --- | --- |
   | `good` | ≤ 1.5 s | < 90° |
   | `fair` | ≤ 5 s | < 360° |
   | `approximate` | ≤ 30 s | – |
   | `poor` | longer, or failed the consistency check | – |

   Backward hand-offs are graded one step lower.

A map becomes a *heading source* for later hand-offs only if it is **healthy** (position and
orientation) and its own hand-off isn't `poor`. A map whose position diverged is never a
source, however well its rates match the gyro *(reviewer finding E7)*. The only exception is a
segment with no healthy map at all, which raises an alarm.

### Step 6: the A.2 world and head frame
- **World:** +Y = (0,0,−1) of the levelled world (gravity-down); +Z = horizontal camera forward,
  averaged over the anchor map's first second; +X = Y × Z (right-handed).
  - If the camera looks almost straight down at the start, "forward" is taken from the camera's
    image-up direction instead.
- **Head frame:** the rectified left camera, `R_world_rect = R_world_imu · R_rectᵀ`, with
  `R_rect = rectified_extrinsics.T_cam0rect_imu[:3,:3]`.
- **Position:** the camera centre, `p_imu + R_world_imu · c`, where `c = −R_rectᵀ t_rect`. It's
  rotated into the A.2 axes and shifted so each map starts at (0,0,0).

### Step 7: write (`deliverable.write`)
- Rows are put back in the original order, and every file in section 3 is written.
- The writer **refuses to write into ORB-SLAM3's own folder.**

## 5. Validation

| Test | Result |
| --- | --- |
| **Synthetic** (`tests/test_reorient.py`): 3 maps, random headings, exact IMU | One rotation fits all maps to within **0.26°**. +Y = gravity. Structure identical. Source folder untouched. |
| **Cut test** (`tools/validate_reorient.py cut`): 70 healthy single-map runs, cut into 2 maps with a random heading and origin | Heading-transfer error for hand-offs graded reliable (median / p95 / worst): **0.37° / 0.86° / 1.2°** at 0.13 s; **0.37° / 1.08° / 1.5°** at 1 s; **0.44° / 1.70° / 2.5°** at 5 s. 30 s gaps: 0.88° / 5.2° (graded `poor`). |
| **Real breaks vs mod-slam** (`tools/validate_reorient.py real`): mod-slam is one continuous map, independent of ORB-SLAM3 | Before the review fixes: graded reliable, median **3.0°** (p90 7.9°); flagged, median 15.5° (max 74°). After them, only 3 hand-offs join two trusted maps: 0.25° (3.8 s), 7.7° (27.6 s), 8.3° (84 s, flagged). |
| **Axes against the footage** | Optical flow in `left_rectified.mp4` matches the gyro mapped into the rectified camera on both image axes. Gravity in the camera frame matches the visible posture. |
| **Batch** (39 multi-map runs, 113 maps, after the review fixes) | Structure identical in 39/39. Export convention `raw_cam0` on all 113 maps. 16 maps healthy; 26 with a reliably shared heading (48 before the stricter tests). Hand-offs: 3 good, 7 fair, 9 approximate, 55 poor. 24 runs alarm, mostly "no healthy map". |

Real breaks happen during violent head motion: 150–2000° of rotation inside the gap. So the
heading across long breaks is only good to a few degrees, and the grade says so.

## 6. `orientation_report.json`

- **Top level:** `conventions`, `frame_note`, `anchor_map`, `healthy_maps`, `unhealthy_maps`,
  `gyro_bias_rad_s`, `imu_offset_s`, `alarms`, `map_files`.
- **`maps[]`:**

  | Field | Meaning |
  | --- | --- |
  | `map_id`, `n_poses`, `t_start`, `t_end` | the map |
  | `healthy`, `health_issues` | health check result and why it failed |
  | `orientation_trusted` | frame guard and gyro correlation both pass |
  | `heading_anchor` | this map defines world +Z |
  | `heading_shared_reliably` | **the field downstream should use** |
  | `yaw_applied_deg` | heading correction applied to the map |
  | `export_convention` | `raw_cam0` / `rectified_cam0` / `unrecognised` |
  | `rate_rel`, `rate_rms_dps`, `rate_corr3` | 3-axis rate test |
  | `gravity_window_p90_deg`, `gravity_window_max_deg` | windowed gravity test |
  | `heading_source` | this map seeded later hand-offs |
  | `tilt_raw_deg`, `tilt_rect_deg`, `gyro_corr`, `speed_p99`, `speed_max` | diagnostics |

- **`bridges[]`:** `from_map`, `to_map`, `direction`, `gap_s`, `bridge_s`,
  `rotation_in_bridge_deg`, `yaw_deg`, `yaw_spread_deg`, `tilt_residual_deg`, `gyro_bias_rad_s`,
  `quality`, `note`.

**Downstream rule of thumb:** use a map's poses if `healthy`. Treat its heading as consistent
with the other maps only if `heading_shared_reliably`.

## 7. Check videos

**Left half:** `left_rectified.mp4` with:
- the world triad drawn 1 m in front of the camera (X red, Y green, Z blue);
- the horizon (yellow);
- the current map and position.

Correct output means:
- across every map switch, the triad keeps pointing the same way in the scene;
- green Y points to gravity-down.

When the wearer looks at their own lap while seated, the camera points past vertical, and image
"down" then legitimately has an upward component.

**Right half:**
- each map from its own origin in the shared axes (top view, X right, Z up on screen);
- ORB-SLAM3's raw maps (each with its own heading);
- a timeline of maps (grey = flagged) with hand-off grades.

| Clip | Maps | Healthy | Anchor | Hand-offs |
| --- | --- | --- | --- | --- |
| PLN-024 003/seg_001 | 2 | 0, 1 | 0 | 0→1 approximate (3.8 s gap; 0.5° vs mod-slam) |
| PIP-305 000/seg_001 | 2 | 0, 1 | 0 | 0→1 fair |
| YTF-753 006/seg_001 | 2 | 1 | 1 | 1→0 fair |
| CAM-352 024/seg_000 | 3 | 2 | 2 | 2→1 fair, 1→0 poor |
| akai-ego-007 001/seg_000 | 4 | 3 | 3 | 3→2 fair, rest poor |
| akai-ego-014 001/seg_000 | 6 | 5 | 5 | 5→4 fair, rest poor |

## 8. Usage

```bash
pip install -r requirements.txt          # numpy; boto3 for S3 input; opencv for --render

# local folders
python -m orbslam3_orientation reorient \
    --orbslam3 <orbslam3 seg dir> --inputs <chunking seg dir> --out <new folder> \
    [--render check.mp4 --video left_rectified.mp4]

# from S3: read-only, output is local
python -m orbslam3_orientation reorient --bucket prod-egc-stereo-v2-data --profile prod \
    --segment <dataset>/.../<chunk>/seg_NNN --out <new folder> [--render check.mp4]

python tests/test_reorient.py
python tools/validate_reorient.py cut  <segdir> ...
python tools/validate_reorient.py real <segdir> ...
```

- **Exit code:** 0 ok; 2 written, but with an alarm.
- **Thresholds:** every one is a flag (`--help`).
- **`process` mode:** experimental; it stitches all maps into one continuous trajectory. It is
  **not** the delivered mode.

## 9. Code layout

| File | Role |
| --- | --- |
| `orbslam3_orientation/io.py` | load ORB-SLAM3, IMU, calibration, time offset (local or S3, read-only) |
| `orbslam3_orientation/geometry.py` | SO(3) helpers, gyro integration, yaw between rotations |
| `orbslam3_orientation/core.py` | `Config` (all thresholds), per-map checks and re-levelling; `process` (stitch mode) |
| `orbslam3_orientation/reorient.py` | **per-map re-orientation:** anchor, gyro bias, hand-offs, A.2 world |
| `orbslam3_orientation/deliverable.py` | writes the ORB-SLAM3-shaped output and `orientation_report.json` |
| `orbslam3_orientation/simulate.py` | reproduces the exporter's per-map levelling, for tests and validation tools |
| `orbslam3_orientation/render.py` | check videos (`render_maps` for reorient) |
| `orbslam3_orientation/__main__.py` | CLI (`reorient`, `process`) |
| `tests/` | synthetic tests |
| `tools/` | real-data validation |

## 10. Limitations

- **Heading across a break** is only as good as the gyro over the gap: sub-degree under 1.5 s,
  several degrees over long gaps with a lot of head rotation. It's graded per hand-off.
- **Flagged (diverged) maps** are re-oriented like the others, but their poses are still
  ORB-SLAM3's failure.
- **Agreement isn't truth.** A wrong calibration would fool ORB-SLAM3, the IMU checks and the
  renders alike.
- **The ORB-SLAM3 stage is inconsistent:** ORB-SLAM3 itself runs with the rectified extrinsic,
  but its exporter uses the raw one. This tool accounts for that. If the exporter is ever
  changed, the frame guard (step 2) will flag it.

## 11. Review findings (2026-10-03) and how they were handled

| # | Finding | Status |
| --- | --- | --- |
| E1 | `bitrobot_to_mcap.load_orbslam3` applies only `R_DOC_OKVIS`, never the camera→IMU rotation, and publishes the result as `ego_imu`. | **Confirmed in the monorepo.** The error is the camera→IMU rotation: 89–91° (akai), 98–106° (bitrobot). Not fixed here: it's monorepo code. Reading this tool's output (already A.2, with markers) avoids it. |
| E2 | Fixing E1 with the rectified extrinsic leaves a residual. | **Confirmed.** The residual equals R1: median 2.5°, 5–95% range 1.0–9.0°, max 9.8°. This tool uses the raw extrinsic for the conversion and the rectified one only for the head frame. |
| E3 | `frame_note` "z is up, gravity-aligned" would be false in stereo mode. | **Mostly not.** In stereo mode the ORB-SLAM3 pipeline exports `pose_frame=camera`, and the exporter levels every map with the accelerometer in both modes. It holds in both; in stereo mode it's one levelling per map, not something the estimator maintains. |
| E4 | `slam_validation.py` reads column 3 as the optical axis (true); the manifest label `imu` is false. | **Confirmed.** Must change together with any E1 fix in the monorepo. |
| E5 | The whole-map gravity check is forced to 0. | **Fixed:** windowed gravity residual (10 s windows, p90 ≤ 2°; needs at least 2 windows), part of orientation trust. |
| E6 | The 0.5° frame guard misses a switch to the rectified extrinsic. | **Fixed:** exact guard at 1e-4°. The rectified convention is identified and handled; anything else is flagged. |
| E7 | `tilt_raw` carries no quality information; the magnitude correlation is marginal; diverged maps seed hand-offs. | **Fixed:** 3-axis rate residual and correlation; heading sources must be healthy. |
| E8 | A loader would re-rotate this output; the output lacked `status.json`. | **Fixed on this side:** every source file is copied, and explicit markers are added in the manifest and `frame_note`. The loader side (dispatch on markers, refuse unmarked files) is monorepo work. |
