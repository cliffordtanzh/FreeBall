import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class BallDetection:
    x: float
    y: float
    radius: float
    frame_idx: int
    confidence: float


class BallDetector:
    """
    Detects volleyball using MOG2 background subtraction + blob filtering.

    Pipeline:
    1. Background subtraction (MOG2) - separate moving objects from static background
    2. Morphological operations - clean noise
    3. Blob detection (contours) - extract connected components
    4. Filtering - size + circularity constraints
    5. Temporal tracking (external) - Kalman filter for consistency

    Rationale: MOG2 is robust for stationary camera. Kalman filter handles disambiguation
    (ball vs player limbs) through motion smoothness.
    """

    def __init__(
        self,
        min_area: int = 50,
        max_area: int = 5000,
        circularity_threshold: float = 0.35,
    ):
        """
        Args:
            min_area: Minimum blob area (px²)
            max_area: Maximum blob area (px²)
            circularity_threshold: Min circularity (0-1, higher = rounder)
        """
        self.min_area = min_area
        self.max_area = max_area
        self.circularity_threshold = circularity_threshold

        # Background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            varThreshold=16,
        )

    def detect(self, frame: np.ndarray) -> list[BallDetection]:
        """
        Detect ball candidates in a single frame.

        Args:
            frame: BGR image

        Returns:
            List of BallDetection objects (before temporal filtering)
        """
        # Step 1: Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Step 2: Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Step 3: Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Size constraint
            if area < self.min_area or area > self.max_area:
                continue

            # Circularity constraint
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.circularity_threshold:
                continue

            # Fitted circle
            (x, y), radius = cv2.minEnclosingCircle(contour)

            detections.append(
                BallDetection(
                    x=x,
                    y=y,
                    radius=radius,
                    frame_idx=-1,
                    confidence=circularity,
                )
            )

        return detections

    def visualize(self, frame: np.ndarray, detections: list[BallDetection]) -> np.ndarray:
        """Draw detections on frame for debugging."""
        vis = frame.copy()
        for det in detections:
            cv2.circle(vis, (int(det.x), int(det.y)), int(det.radius), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"C:{det.confidence:.2f}",
                (int(det.x) - 40, int(det.y) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
        return vis

    def visualize_tracked(self, frame: np.ndarray, tracked_balls: list) -> np.ndarray:
        """Draw tracked balls (from Kalman filter) on frame."""
        vis = frame.copy()
        for ball in tracked_balls:
            # Draw circle
            cv2.circle(vis, (int(ball.x), int(ball.y)), int(ball.radius), (0, 255, 0), 2)

            # Draw velocity vector
            vx_px, vy_px = int(ball.vx * 3), int(ball.vy * 3)  # Scale for visibility
            cv2.arrowedLine(
                vis,
                (int(ball.x), int(ball.y)),
                (int(ball.x) + vx_px, int(ball.y) + vy_px),
                (255, 0, 0),
                2,
            )

            # Draw stats
            text = f"Age:{ball.age} C:{ball.confidence:.2f}"
            cv2.putText(
                vis,
                text,
                (int(ball.x) - 60, int(ball.y) - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
        return vis

    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return MOG2 foreground mask."""
        return self.bg_subtractor.apply(frame)
