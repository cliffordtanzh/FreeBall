import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackedBall:
    """Kalman-filtered ball state."""
    x: float
    y: float
    vx: float  # velocity x
    vy: float  # velocity y
    radius: float
    age: int  # frames since creation
    confidence: float
    last_detection_frame: int


class KalmanTracker:
    """
    Kalman filter for tracking volleyball across frames.

    State: [x, y, vx, vy] (position + velocity)
    Measurement: [x, y] (detection position)
    """

    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 5.0):
        """
        Args:
            process_noise: Process noise (position drift)
            measurement_noise: Measurement noise (detection uncertainty)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def create_filter(self, x: float, y: float) -> dict:
        """Create new Kalman filter state for ball at (x, y)."""
        return {
            "state": np.array([[x], [y], [0.0], [0.0]], dtype=np.float32),  # [x, y, vx, vy]
            "covariance": np.eye(4, dtype=np.float32) * 10.0,
        }

    def predict(self, filter_state: dict, dt: float = 1.0) -> np.ndarray:
        """Predict next position given velocity."""
        # State transition: x_new = x + vx*dt, y_new = y + vy*dt
        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        Q = np.eye(4, dtype=np.float32) * self.process_noise

        filter_state["state"] = A @ filter_state["state"]
        filter_state["covariance"] = A @ filter_state["covariance"] @ A.T + Q

        return filter_state["state"]

    def update(self, filter_state: dict, measurement: np.ndarray) -> np.ndarray:
        """Update state given detection measurement [x, y]."""
        # Measurement matrix: observe only [x, y]
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        R = np.eye(2, dtype=np.float32) * self.measurement_noise

        # Innovation
        y = measurement - H @ filter_state["state"]

        # Innovation covariance
        S = H @ filter_state["covariance"] @ H.T + R

        # Kalman gain
        K = filter_state["covariance"] @ H.T @ np.linalg.inv(S)

        # Update state
        filter_state["state"] = filter_state["state"] + K @ y

        # Update covariance
        filter_state["covariance"] = (np.eye(4, dtype=np.float32) - K @ H) @ filter_state["covariance"]

        return filter_state["state"]

    def mahalanobis_distance(self, filter_state: dict, measurement: np.ndarray) -> float:
        """Distance metric for data association."""
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        R = np.eye(2, dtype=np.float32) * self.measurement_noise
        innovation = measurement - H @ filter_state["state"]
        S = H @ filter_state["covariance"] @ H.T + R

        try:
            dist = np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation)
            return float(dist[0, 0])
        except np.linalg.LinAlgError:
            return float('inf')


class BallTracker:
    """
    Track detected balls across frames using Kalman filtering.

    Handles:
    - Track creation from new detections
    - Track prediction and update
    - Data association (matching detections to tracks)
    - Track deletion (old/lost tracks)
    """

    def __init__(
        self,
        max_track_age: int = 30,  # frames
        max_association_distance: float = 50.0,  # Mahalanobis distance
        min_track_age_for_output: int = 2,  # require ≥2 frames of tracking
    ):
        self.kalman = KalmanTracker()
        self.tracks: dict[int, dict] = {}  # track_id -> track_state
        self.next_track_id = 0
        self.max_track_age = max_track_age
        self.max_association_distance = max_association_distance
        self.min_track_age_for_output = min_track_age_for_output

    def update(self, detections: list, frame_idx: int) -> list[TrackedBall]:
        """
        Update tracks with new detections.

        Args:
            detections: List of (x, y, radius, confidence) tuples
            frame_idx: Current frame number

        Returns:
            List of confirmed tracked balls
        """
        # Predict all tracks
        for track_id in list(self.tracks.keys()):
            self.kalman.predict(self.tracks[track_id]["filter"])

        # Data association: match detections to tracks
        associated = set()
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]

            # Find closest detection
            best_det_idx = -1
            best_distance = self.max_association_distance

            for det_idx, (x, y, radius, confidence) in enumerate(detections):
                if det_idx in associated:
                    continue

                measurement = np.array([[x], [y]], dtype=np.float32)
                distance = self.kalman.mahalanobis_distance(track["filter"], measurement)

                if distance < best_distance:
                    best_distance = distance
                    best_det_idx = det_idx

            # Update track if match found
            if best_det_idx >= 0:
                x, y, radius, confidence = detections[best_det_idx]
                measurement = np.array([[x], [y]], dtype=np.float32)
                self.kalman.update(track["filter"], measurement)
                track["age"] += 1
                track["last_detection_frame"] = frame_idx
                track["radius"] = radius  # Update radius estimate
                track["confidence"] = confidence
                associated.add(best_det_idx)
            else:
                # No match: decay track
                track["age"] += 1

            # Delete old tracks
            if track["age"] > self.max_track_age:
                del self.tracks[track_id]

        # Create new tracks for unassociated detections
        for det_idx, (x, y, radius, confidence) in enumerate(detections):
            if det_idx not in associated:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    "filter": self.kalman.create_filter(float(x), float(y)),
                    "age": 1,
                    "last_detection_frame": frame_idx,
                    "radius": radius,
                    "confidence": confidence,
                }

        # Output confirmed tracks
        output = []
        for track_id, track in self.tracks.items():
            if track["age"] >= self.min_track_age_for_output:
                state = track["filter"]["state"]
                output.append(
                    TrackedBall(
                        x=float(state[0, 0]),
                        y=float(state[1, 0]),
                        vx=float(state[2, 0]),
                        vy=float(state[3, 0]),
                        radius=track["radius"],
                        age=track["age"],
                        confidence=track["confidence"],
                        last_detection_frame=track["last_detection_frame"],
                    )
                )

        return output
