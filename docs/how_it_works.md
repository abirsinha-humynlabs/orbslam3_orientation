# ORB-SLAM3 orientation post-process: how it works

**Code:** `s3://stage-humyn-egocentric-stereo-data/labelling_results/orb_orientation/code/`
(package in `src/`, version 1.3.0)

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

A folder that mirrors ORB-SLAM3's. **The default convention (`--convention mcap`) needs no
downstream code change:** `trajectory.npz` holds exactly what the MCAP exporter's
`load_orbslam3` assumes, so its existing fixed −90°-about-X rotation (`R_DOC_OKVIS`) produces
Opeth A.2. This was verified by running the unmodified monorepo loader on the output
(`tools/check_with_mcap_loader.py`, section 5).

| File | Content |
| --- | --- |
| `trajectory.npz` | **Same keys, dtypes, row count and row order.** `timestamp_s`, `segment` and `segment_files` are identical. `rotation` and `position_m` are re-oriented (table below). `frame_note` starts with `ORB_ORIENTATION imu_zup`. |
| `segments/segNN_mapK_poses.txt` | Same names and columns (`t x y z r00..r22`, `%.9f`), same convention as `trajectory.npz`. New header. |
| `trajectory_a2_camera.npz` | **Sidecar:** the same rows, already in Opeth A.2, as the rectified left camera's pose. For readers that want A.2 directly; the MCAP exporter ignores it. `frame_note` starts with `OPETH_A2`. |
| `segments_manifest.json` | Copied. Per-map extents recomputed. **Markers:** `pose_frame` / `rotation_frame` / `position_frame` = `imu` (now true), `world_frame` = `gravity_zup_x_right_y_forward`, `axis_convention` = `zup_then_rx_minus90_to_opeth_a2`, `gravity_axis` = `+z_up`. A `post_process` block is added. |
| `report.json`, `status.json`, … | **Every other file** in ORB-SLAM3's folder is copied unchanged, so the loader finds everything it expects *(E8)*. |
| `orientation_report.json` | **New:** conventions, per-map health, heading hand-offs, alarms (section 6). |

What the arrays mean:

| | `trajectory.npz` (default, "mcap") | `trajectory_a2_camera.npz` (sidecar) |
| --- | --- | --- |
| world axes | gravity-aligned **+x right, +y forward, +z up**; the same heading in EVERY map | **Opeth A.2: +X right, +Y down, +Z forward**; the same heading in every map |
| relation | A.2 = R_x(−90°) · this, which is what `load_orbslam3` applies | – |
| `rotation` | world ← **IMU** body | world ← **rectified left camera** (`left_rectified.mp4`: +X image right, +Y image down, +Z optical axis) |
| `position_m` | IMU origin | rectified left camera centre |
| origin | **each map its own: (0,0,0) at its first pose** | same |
| `timestamp_s` | unchanged (camera clock); IMU clock = `timestamp_s + measured_imu_time_offset_s` (the loader adds it) | same |

"Forward" (+y in the default file, +Z in A.2) is the operator's horizontal facing direction
during the first second of the *heading-anchor map* (the first healthy map; usually map 0).

`--convention a2-camera` swaps the two: `trajectory.npz` is then the A.2 camera version, with
markers `axis_convention: opeth_a2` (a reader must not rotate it again), and the sidecar is
`trajectory_imu_zup.npz`.

**No stitching.** No pose is added, removed or moved between maps. Maps stay separate, as
ORB-SLAM3 delivered them. Only the axes change, and all maps now share them.

**Downstream:** only configuration changes, so that the exporter reads this folder instead of
ORB-SLAM3's:
- `--vio <path>` for a single take; or
- publish under the same layout (`orbslam3/<…>/segments/<seg>/`) in a bucket and pass
  `--orbslam-bucket <bucket>`. That flag is also where the exporter looks for OKVIS output.

Do **not** overwrite ORB-SLAM3's originals: the SLAM selector and the monorepo's
`analysis/build_dataset.py` read those rotations as the raw camera.

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

- **Export-convention guard, decided once per segment** (an exporter run uses one extrinsic for
  every map). The exporter levels each map with `mean(R_wb · acc)`, so with the extrinsic it
  really used, the residual tilt is **exactly 0** (below 1e-6° on 133 maps). The other extrinsic
  gives at least 0.0045°.

  | Result | When | Action |
  | --- | --- | --- |
  | `raw_cam0` | every map exactly 0 (≤ 1e-4°) with the raw extrinsic: today's exporter | `R_world_imu = R_npz · T_cam0_imu[:3,:3]` |
  | `rectified_cam0` | every map exactly 0 with the rectified extrinsic: the exporter switched | used, alarm raised |
  | `raw_cam0_approx` / `rectified_cam0_approx` | not exact (the exporter's numerics changed), but the pooled RMS residual of one extrinsic is ≤ 0.05° **and** at least 3× smaller than the other's | used, orientation still trusted, alarm "identified only approximately" |
  | `unrecognised` | anything else | no map trusted, alarm raised |

  - A 0.5° tolerance can't tell the two apart: 49 of 132 maps have a rectified-extrinsic residual
    below 0.5° *(reviewer finding E6)*.
  - The exact test alone breaks on any upstream change in numerics, failing every map
    *(second review, N1)*; the margin fallback handles that.
  - It is never a bare argmin: with a changed time alignment, raw reads up to 0.24° while
    rectified reads as low as 0.018°, so close calls stay `unrecognised`.
  - Deciding per segment stops one odd map from switching convention on its own (seen in the
    synthetic test).
- **Re-level:** the smallest rotation that puts the measured gravity exactly on +z.
- **Position health:** at least 3 s long; speed at 10 Hz p99 ≤ 3 m/s and max ≤ 15 m/s.
- **Orientation tests**, independent of position:

  | Test | Limit | Calibrated on 145 healthy / 102 diverged maps | Finding |
  | --- | --- | --- | --- |
  | 3-axis rate residual: RMS of (trajectory body rate − (gyro − bias)), relative to the gyro's RMS rate | ≤ 0.25 | healthy median 0.06, p95 0.20; diverged median 0.36 | E7 |
  | correlation of the 3 rate components | ≥ 0.95 | healthy p5 0.98 | E7 |
  | windowed gravity residual: mean specific force per 10 s window vs the vertical, p90 over windows | ≤ 2° | healthy p99 1.7; diverged median 2.1 | E5 |

  Neither test is decisive alone: about half of the diverged maps pass each one by itself.
  Together, and combined with the position checks, they separate well *(second review, N4)*.
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

### Step 6: the operator world, the head frame and the delivered convention
- **World:** +Y = (0,0,−1) of the levelled world (gravity-down); +Z = horizontal camera forward,
  averaged over the anchor map's first second; +X = Y × Z (right-handed).
  - If the camera looks almost straight down at the start, "forward" is taken from the camera's
    image-up direction instead.
- **Head frame:** the rectified left camera, `R_world_rect = R_world_imu · R_rectᵀ`, with
  `R_rect = rectified_extrinsics.T_cam0rect_imu[:3,:3]`.
- **Position:** the camera centre, `p_imu + R_world_imu · c`, where `c = −R_rectᵀ t_rect`. It's
  rotated into the A.2 axes and shifted so each map starts at (0,0,0).

- **Delivered convention (default):** the IMU pose (`R_world_imu`, IMU origin) in the Z-up
  operator world: +x right, +y forward, +z up, i.e. R_x(+90°) · A.2. The MCAP loader's fixed
  R_x(−90°) then gives exactly A.2. The A.2 camera pose goes to the sidecar.

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
| **Unmodified MCAP loader** (`tools/check_with_mcap_loader.py`): the monorepo's `bitrobot_to_mcap.load_orbslam3` run on the default output of the six rendered clips | All six PASS: the published pose, with the camera calibration applied, equals the A.2 camera sidecar to 0.002°; gravity → −Y (windowed p90 0.3–1.1° on healthy maps); the starting facing → +Z (X ≤ 0.02). |
| **Original vs post-processed through the exporter** (`tools/compare_original_vs_post.py`): both folders through the unmodified `load_orbslam3`; the published "`ego_imu`" orientation tested against the IMU's own gyro and accelerometer | Original: published body is the raw camera (0.2–1.6° from it), **90–104° off the real IMU**, matching the calibration angle to 0.4° (E1). Post-processed: **0.2–1.6° off the IMU**, the same leftover as ORB-SLAM3's own rotation-vs-gyro agreement. Gravity (healthy maps) 31–105° → 0.2–0.6°. |
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
python -m src reorient \
    --orbslam3 <orbslam3 seg dir> --inputs <chunking seg dir> --out <new folder> \
    [--render check.mp4 --video left_rectified.mp4]

# from S3: read-only, output is local
python -m src reorient --bucket prod-egc-stereo-v2-data --profile prod \
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
| `src/io.py` | load ORB-SLAM3, IMU, calibration, time offset (local or S3, read-only) |
| `src/geometry.py` | SO(3) helpers, gyro integration, yaw between rotations |
| `src/core.py` | `Config` (all thresholds), per-map checks and re-levelling; `process` (stitch mode) |
| `src/reorient.py` | **per-map re-orientation:** anchor, gyro bias, hand-offs, A.2 world |
| `src/deliverable.py` | writes the ORB-SLAM3-shaped output and `orientation_report.json` |
| `tools/check_with_mcap_loader.py` | runs the unmodified monorepo MCAP loader on an output folder and checks what it publishes |
| `tools/compare_original_vs_post.py` | runs the loader on ORB-SLAM3's original folder and on the post-processed folder; tests both published orientations against the IMU's own sensors |
| `src/simulate.py` | reproduces the exporter's per-map levelling, for tests and validation tools |
| `src/render.py` | check videos (`render_maps` for reorient) |
| `src/__main__.py` | CLI (`reorient`, `process`) |
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
| E1 | `bitrobot_to_mcap.load_orbslam3` applies only `R_DOC_OKVIS`, never the camera→IMU rotation, and publishes the result as `ego_imu`. | **Confirmed in the monorepo** (89–91° on akai, 98–106° on bitrobot, from 443 calibrations; measured through the loader on six clips: 90.6–104.3°, `tools/compare_original_vs_post.py`). **Avoided without a monorepo change (v1.3.0):** the default output holds the IMU pose in the Z-up world the loader assumes, so its rotation yields correct A.2. Anyone who exports ORB-SLAM3's original output still gets the bug. |
| E2 | Fixing E1 with the rectified extrinsic leaves a residual. | **Confirmed.** The residual equals R1: median 2.5°, 5–95% range 1.0–9.0°, max 9.8°. This tool uses the raw extrinsic for the conversion and the rectified one only for the head frame. |
| E3 | `frame_note` "z is up, gravity-aligned" would be false in stereo mode. | **Mostly not.** In stereo mode the ORB-SLAM3 pipeline exports `pose_frame=camera`, and the exporter levels every map with the accelerometer in both modes. It holds in both; in stereo mode it's one levelling per map, not something the estimator maintains. |
| E4 | `slam_validation.py` reads column 3 as the optical axis (true); the manifest label `imu` is false. | **Confirmed.** Must change together with any E1 fix in the monorepo. |
| E5 | The whole-map gravity check is forced to 0. | **Fixed:** windowed gravity residual (10 s windows, p90 ≤ 2°; needs at least 2 windows), part of orientation trust. |
| E6 | The 0.5° frame guard misses a switch to the rectified extrinsic. | **Fixed:** exact guard at 1e-4°. The rectified convention is identified and handled; anything else is flagged. |
| E7 | `tilt_raw` carries no quality information; the magnitude correlation is marginal; diverged maps seed hand-offs. | **Fixed:** 3-axis rate residual and correlation; heading sources must be healthy. |
| E8 | A loader would re-rotate this output; the output lacked `status.json`. | **Fixed:** every source file is copied. Since v1.3.0 the default output is *meant* to be rotated by the loader (Z-up IMU convention), so no loader change is needed. Markers describe the convention; the `a2-camera` convention is still marked "do not rotate". |

### Second review (2026-10-03): N1–N4

| # | Finding | Status |
| --- | --- | --- |
| N1 | The exact export-convention guard is brittle: any change in the exporter's numerics would make every map "unrecognised". | **Fixed.** Margin-based fallback (`*_approx`: pooled residual ≤ 0.05° and ≥ 3× smaller than the other), decided once per segment; ambiguous cases stay `unrecognised`. Tested: exact, approximate and ambiguous. A float32 round-trip of the rotations stays exact (< 1e-7°). |
| N2 | `process()` still had the whole-trajectory gravity check (0 by construction), and a test that could not fail. | **Fixed.** Windowed per-map check (10 s windows, ≥ 2 windows, worst map's p90), plus a test with a drifting map that must trip it. |
| N3 | npz files left open (`NpzFile` keeps its zip handle), so the tests failed on Windows. | **Fixed.** `io.read_npz` loads into a dict and closes the file; used in `src`, `tests` and `tools`. |
| N4 | Thin threshold margins; a comment said healthy "p97 ~0.25". | **Documented**: the tests aren't decisive alone. Comment corrected: healthy p95 0.20, p97 0.23. |
