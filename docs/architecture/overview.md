# FreeBall Project Architecture

## Overview

FreeBall is a volleyball analysis tool built to automatically detect rallies and propose clips from YouTube videos.

**Tech Stack**: Python (backend), React (frontend), FastAPI, OpenCV, PyTorch

**Goal**: Prototype → PoC → MVP progression

---

## Project Structure

```
FreeBall/
├── backend/              # FastAPI backend + ML logic
│   ├── scripts/          # Standalone test/experiment scripts
│   └── src/
│       ├── main.py       # FastAPI app entry
│       ├── controllers/  # HTTP endpoints
│       ├── functionality/
│       │   ├── ball-detection/         # Ball localization via motion + circularity
│       │   ├── rally-detection/        # Rally segmentation using ball trajectory
│       │   ├── player-detection/       # Player localization & pose
│       │   └── action-detection/       # Spike, block, receive recognition
│       ├── routes/       # API routes
│       ├── services/     # Business logic
│       ├── middlewares/  # Auth, logging, etc
│       └── utils/        # Helpers (video_utils, math, etc)
├── frontend/             # React app
├── data/
│   ├── raw/              # Downloaded YouTube videos (~263MB, 5 samples)
│   ├── processed/        # Preprocessed frames, detections
│   └── schemas/          # Data validation schemas
├── docs/
│   ├── architecture/     # System design docs
│   ├── decisions/        # ADRs (architecture decision records)
│   └── prd/              # Product requirements
├── literature/           # 25+ research papers (Tier 1 & 2)
└── pyproject.toml        # Python dependencies

```

---

## Module Dependency Graph

```
Ball Detection
  ↓
Rally Detection (consumes ball detections)
  ↓
Player Detection (parallel with Rally)
  ↓
Action Detection (consumes player poses + rally context)
```

Each module is independent but can be composed. Ball detection is foundational.

---

## Current Phase: Prototype

**Objective**: Get ball detection + rally segmentation working end-to-end

### Stage 1: Ball Detection (This Weekend)
- **Approach**: Motion-based (frame differencing) + circularity filtering
- **Why**: Fast, interpretable, doesn't require training data
- **Challenge**: Ball + player motion blur when close → circularity constraint filters player limbs
- **Validation**: Interactive frame stepping with `test_ball_detection.py` script
- **Fallback**: Background subtraction (MOG2) or optical flow if needed

### Stage 2: Rally Segmentation (Next)
- **Input**: Ball detections from Stage 1
- **Implementation**: Kalman filtering + parabola fitting
- **Logic**: Serve trajectory (start) → active flight → ground impact (end)
- **Validation**: Test on 5 curated videos (indoor/outdoor, casual/competitive, multiple angles)

### Stage 3: Player Detection (Later)
- **Goal**: Localize all players, infer positions
- **Method**: YOLO or lightweight pose model
- **For rally end**: Pose estimation (supporting signal—celebrations, relaxed posture)

---

## Key References

**Ball Tracking Papers** (literature/ball_tracking_and_trajectory_analysis.md):
- Chen 2012: Kalman filtering for ball tracking
- Liu 2019: Deep learning trajectory prediction
- Multi-object tracking under occlusion

**Architecture Decision**: [[rally-detection-architecture-decision]]
- Ball trajectory = necessary condition for rally
- Pose estimation = sufficient condition for rally END (when ball trajectory ambiguous)

---

## Next Steps

1. Run `test_ball_detection.py` on first video
   - Frame-by-frame interactive stepping
   - Tune circularity threshold, blur kernel based on results
   - Save annotated frames to `data/processed/`

2. Implement Kalman tracker in `rally-detection/`
   - Track ball centroids across frames
   - Predict next position based on velocity

3. Add trajectory analysis (parabola fitting) to detect rally boundaries
   - Serve: upward arc with high velocity
   - Ground hit: ball stops moving

4. Wire to FastAPI endpoint for video upload + processing

---

## Development Notes

- **Determinism**: Fixed seeds, reproducible results on same data
- **Experiments**: Train set (videos 1-3) vs held-out set (videos 4-5)
- **Results**: Write to `data/processed/` + markdown summary in `docs/`
- **No training data**: Prototype uses classical CV, not deep learning
- **Visualization**: Use `cv2.imshow()` with spacebar navigation (key = cv2.waitKey(-1))

---

**Last Updated**: 2026-10-24  
**Author**: Clifford + Claude

