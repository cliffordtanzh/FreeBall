# Ball Detection: MOG2 + Kalman Tracking

## Summary

**Previous approach failed**: Optical flow magnitude was consistently < 2.0 px/frame, so motion thresholding killed all detections. Back to basics.

**New approach**: MOG2 foreground detection + Kalman filter tracking.

1. **MOG2 background subtraction** — learns background model, outputs moving foreground
2. **Blob detection** (size + circularity) — extract candidates from foreground
3. **Kalman filter tracking** — temporal smoothing across frames
   - Discriminator: **motion smoothness** (ball = parabolic arc, limbs = noisy/jittery)
   - Data association: nearest-neighbor matching with Mahalanobis distance
   - Track confirmation: require ≥2 frames to reject single-frame noise

## Architecture

```
Frame 1 → MOG2 (foreground) → Blob detection → [x, y, radius] → Kalman tracker
Frame 2 → MOG2 (foreground) → Blob detection → [x, y, radius] → Kalman tracker
...                                                               ↓
                                                    Confirmed tracks (smooth motion)
```

## Key Components

**`detector.py`**:
- `BallDetector.detect()` — MOG2 + contour extraction (frame-level)
- Returns all blob candidates (some are limbs, some are ball)

**`tracker.py`**:
- `KalmanTracker` — Kalman filter math (predict + update)
- `BallTracker` — Track management, data association, confirmation
  - Predicts ball position each frame
  - Matches detections to tracks (nearest neighbor, Mahalanobis distance)
  - Creates/deletes tracks
  - Outputs only confirmed tracks (age ≥ 2 frames)

## Filtering Strategy

**Frame-level** (detector.py):
- MOG2 foreground
- Size: 50-5000 px²
- Circularity: ≥ 0.35

**Temporal-level** (tracker.py):
- **Motion smoothness**: Kalman filter maintains velocity; smooth arcs (ball) have low innovation, jittery motion (limbs) have high innovation
- **Track age**: Require ≥2 frames of consistent matching before output (rejects single-frame noise)
- **Data association**: Mahalanobis distance — tracks with low covariance are more selective

## How to Test

```bash
conda activate FreeBall
python backend/scripts/test_ball_detection.py "data/raw/Game 2 [0zgWIjgqVcA].mp4"
```

Controls:
- **SPACEBAR**: Next frame
- **'d'**: Toggle between raw detections (MOG2) and tracked balls (Kalman)
- **'q'**: Quit

## Visualization

**OFF (default)**: Tracked balls
- Green circle = confirmed ball (age ≥ 2 frames)
- Blue arrow = velocity vector (should be smooth parabolic trajectory)
- Stats: Age (frames tracked), Confidence (circularity), Track count

**ON ('d' key)**: Raw detections
- Green circle = blob candidate from MOG2
- All blobs including player limbs, clothes
- Use this to tune circularity threshold

## Tuning

**Too many false positives (player limbs detected as tracked balls)**:
1. Increase `circularity_threshold` in detector.py (e.g., 0.40, 0.45) — elbows/knees are less circular
2. Increase `min_track_age_for_output` in tracker.py (e.g., 3, 4) — limbs don't track smoothly

**Missing the ball**:
1. Decrease `circularity_threshold` (e.g., 0.30) — ball may be slightly deformed from rotation
2. Decrease `max_association_distance` in tracker.py (e.g., 30, 40) — too strict can lose tracks
3. Adjust `min_area`/`max_area` if ball appears too small/large

## Expected Behavior

- **Limbs**: MOG2 picks them up (frame-level), but Kalman filter rejects them (jittery motion, no smooth parabolic arc)
- **Ball**: MOG2 picks it up, Kalman filter confirms it (smooth trajectory consistent with physics)
- **Result**: Tracked balls = mostly ball, few false positives

---

**Last Updated**: 2026-10-24  
**Approach**: MOG2 foreground detection + Kalman filter tracking
**Status**: Ready for field testing
