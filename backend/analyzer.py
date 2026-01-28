"""
Video processing module for golf swing analysis.
Extracts pose landmarks from video frames using MediaPipe Pose.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from typing import Optional
import os


# ============================================================================
# CONSTANTS
# ============================================================================

MIN_LANDMARK_CONFIDENCE = 0.5
MIN_VALID_FRAMES_RATIO = 0.5

# MediaPipe pose landmark indices
# Required landmarks - frame rejected if any of these are below threshold
REQUIRED_LANDMARKS = {
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_hip': 23,
    'right_hip': 24,
}

# Optional landmarks - included if available, but don't reject frame if missing
OPTIONAL_LANDMARKS = {
    'left_wrist': 15,
    'right_wrist': 16,
}

# Path to the pose landmarker model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pose_landmarker.task')


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class VideoProcessingError(Exception):
    """Raised when video cannot be read or is corrupted."""
    pass


class NoPoseDetectedError(Exception):
    """Raised when no person is detected in the video."""
    pass


class LowConfidenceError(Exception):
    """Raised when pose detection confidence is too low."""
    pass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_landmarks(pose_landmarks, frame_width: int, frame_height: int) -> Optional[dict]:
    """
    Extract landmarks from MediaPipe pose results.

    Returns dict with landmark data or None if any required landmark
    has visibility below threshold. Optional landmarks are included
    if available but don't cause rejection.
    """
    landmarks = {}

    # Check required landmarks - reject frame if any are below threshold
    for name, landmark_id in REQUIRED_LANDMARKS.items():
        lm = pose_landmarks[landmark_id]

        if lm.visibility < MIN_LANDMARK_CONFIDENCE:
            return None

        landmarks[name] = {
            'x': lm.x,
            'y': lm.y,
            'z': lm.z,
            'visibility': lm.visibility,
            'pixel_x': int(lm.x * frame_width),
            'pixel_y': int(lm.y * frame_height)
        }

    # Add optional landmarks if they meet threshold
    for name, landmark_id in OPTIONAL_LANDMARKS.items():
        lm = pose_landmarks[landmark_id]

        if lm.visibility >= MIN_LANDMARK_CONFIDENCE:
            landmarks[name] = {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility,
                'pixel_x': int(lm.x * frame_width),
                'pixel_y': int(lm.y * frame_height)
            }

    return landmarks


def _calculate_frame_confidence(landmarks: dict) -> float:
    """Calculate average visibility across all landmarks in a frame."""
    total = sum(lm['visibility'] for lm in landmarks.values())
    return total / len(landmarks)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def analyze_swing(video_path: str, frame_skip: int = 2) -> dict:
    """
    Analyze a golf swing video and extract pose landmarks.

    Args:
        video_path: Path to the video file
        frame_skip: Process every Nth frame (default 2)

    Returns:
        dict with keys:
            - frames_data: list of frame dicts
            - video_info: dict with fps, total_frames, duration
            - avg_confidence: float

    Raises:
        VideoProcessingError: For invalid/corrupted video
        NoPoseDetectedError: No person detected in video
        LowConfidenceError: Pose confidence too low
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise VideoProcessingError(
            "Unable to open video file. The file may be corrupted or in an unsupported format."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0 or fps == 0:
        cap.release()
        raise VideoProcessingError(
            "Unable to read video metadata. The file may be corrupted."
        )

    duration = total_frames / fps

    # Create pose landmarker with video mode
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    frames_data = []
    frame_index = 0
    valid_frame_count = 0
    total_confidence = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_skip != 0:
                frame_index += 1
                continue

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Calculate timestamp in milliseconds
            timestamp_ms = int(frame_index * 1000 / fps)

            # Detect pose
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = _extract_landmarks(
                    result.pose_landmarks[0],
                    frame_width,
                    frame_height
                )

                if landmarks is not None:
                    confidence = _calculate_frame_confidence(landmarks)
                    frames_data.append({
                        'frame_index': frame_index,
                        'landmarks': landmarks,
                        'confidence': confidence
                    })
                    valid_frame_count += 1
                    total_confidence += confidence

            frame_index += 1
    finally:
        cap.release()
        landmarker.close()

    if valid_frame_count == 0:
        raise NoPoseDetectedError(
            "No person detected in the video. Please ensure you are fully visible in the frame."
        )

    processed_frame_count = (total_frames + frame_skip - 1) // frame_skip
    valid_ratio = valid_frame_count / processed_frame_count

    if valid_ratio < MIN_VALID_FRAMES_RATIO:
        raise LowConfidenceError(
            f"Pose detection confidence too low. Only {valid_ratio:.0%} of frames had reliable detection. "
            "Please ensure good lighting and that your full body is visible."
        )

    avg_confidence = total_confidence / valid_frame_count

    return {
        'frames_data': frames_data,
        'video_info': {
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'frame_width': frame_width,
            'frame_height': frame_height,
            'frame_skip': frame_skip,
            'processed_frames': len(frames_data)
        },
        'avg_confidence': avg_confidence
    }
