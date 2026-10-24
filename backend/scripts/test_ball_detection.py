"""
Interactive ball detection + tracking testing script.

Uses MOG2 foreground detection + Kalman filter tracking.

Usage:
    python backend/scripts/test_ball_detection.py <video_path> [--debug]

Controls:
    SPACEBAR: Next frame
    'd':      Toggle debug mode (shows raw detections vs tracked)
    'q':      Quit

Parameters (tune in detector.py):
    circularity_threshold: Min circularity (default: 0.35, range 0-1)
    min_area / max_area: Blob size constraints (default: 50-5000 px²)

Parameters (tune in tracker.py):
    max_association_distance: Max distance to associate detection to track (default: 50 px)
    min_track_age_for_output: Min frames before track is confirmed (default: 2)
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.functionality.ball_detection.detector import BallDetector
from src.functionality.ball_detection.tracker import BallTracker


def test_ball_detection(video_path: str, show_debug: bool = False):
    """Test ball detection + tracking on video with interactive frame stepping."""
    detector = BallDetector(
        min_area=50,
        max_area=5000,
        circularity_threshold=0.35,  # Tune: lower = accept less circular shapes
    )
    tracker = BallTracker(
        max_track_age=30,
        max_association_distance=50.0,
        min_track_age_for_output=2,
    )

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    debug_mode = show_debug

    print(f"Loaded: {video_path}")
    print("Controls: SPACEBAR=next, 'd'=toggle debug, 'q'=quit")
    print(f"Debug mode: {'ON (raw detections)' if debug_mode else 'OFF (tracked balls)'}")
    print()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Detect ball candidates
        detections = detector.detect(frame)

        # Step 2: Update tracker
        tracked_balls = tracker.update(
            [(d.x, d.y, d.radius, d.confidence) for d in detections],
            frame_idx,
        )

        # Visualize
        if debug_mode:
            # Show raw detections from MOG2
            viz = detector.visualize(frame, detections)
            mode_text = "Mode: RAW DETECTIONS (from MOG2)"
        else:
            # Show Kalman-tracked balls
            viz = detector.visualize_tracked(frame, tracked_balls)
            mode_text = "Mode: TRACKED BALLS (Kalman filtered)"

        # Stats
        h_viz, w_viz = viz.shape[:2]
        text = f"Frame: {frame_idx} | Raw: {len(detections)} | Tracked: {len(tracked_balls)} | Tracks: {len(tracker.tracks)}"
        cv2.putText(viz, text, (10, h_viz - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(viz, mode_text, (10, h_viz - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Ball Detection + Tracking", viz)
        frame_idx += 1

        # Wait for key
        key = cv2.waitKey(-1)
        if key == ord("q"):
            break
        elif key == ord("d"):
            debug_mode = not debug_mode
            mode = "RAW DETECTIONS" if debug_mode else "TRACKED BALLS"
            print(f"Switched to: {mode}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    video_path = sys.argv[1]
    debug = "--debug" in sys.argv
    test_ball_detection(video_path, show_debug=debug)
