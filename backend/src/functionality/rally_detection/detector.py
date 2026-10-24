import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional

from backend.src.functionality.ball_detection.detector import BallDetector, BallDetection


@dataclass
class RallySegment:
    start_frame: int
    end_frame: int
    confidence: float


class RallyDetector:
    """
    Segments rallies using ball trajectory as primary signal.

    Pipeline:
    1. Ball detection (via BallDetector service)
    2. Ball tracking (Kalman filter)
    3. Trajectory analysis (physics-based parabola fitting)
    4. Rally segmentation (start/end detection from trajectory)
    """

    def __init__(self, fps: int = 30, court_height: float = 9.0):
        self.fps = fps
        self.court_height = court_height
        self.ball_detector = BallDetector()
        self.ball_history = []

    def detect_rallies(self, video_path: str) -> list[RallySegment]:
        """Process video and return rally boundaries."""
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        rallies = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Use ball detector service
            detections = self.ball_detector.detect(frame)
            for det in detections:
                det.frame_idx = frame_idx

            # TODO: Track and segment
            self.ball_history.extend(detections)

            frame_idx += 1

        cap.release()
        return rallies

    def _analyze_trajectory(self) -> list[RallySegment]:
        """Analyze ball trajectory to detect serve (start) and ground hit (end)."""
        # TODO: Implement Kalman filtering + parabola fitting
        return []
