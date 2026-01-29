"""
Visualization tool for debugging pose detection.
Generates an annotated video with pose landmarks and skeleton overlay.

Usage:
    python visualize.py input_video.mp4
    python visualize.py input_video.mp4 --output annotated.mp4
"""

import argparse
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import os

from analyzer import MODEL_PATH, MIN_LANDMARK_CONFIDENCE

# Landmark indices (matching analyzer.py)
LANDMARKS = {
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_hip': 23,
    'right_hip': 24,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
}

# Skeleton connections to draw
SKELETON_CONNECTIONS = [
    ('left_shoulder', 'right_shoulder'),
    ('left_hip', 'right_hip'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_hip', 'left_knee'),
    ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'),
    ('right_knee', 'right_ankle'),
]

# Colors (BGR format)
COLOR_LANDMARK = (0, 255, 0)       # Green for valid landmarks
COLOR_LOW_CONF = (0, 165, 255)     # Orange for low confidence
COLOR_SKELETON = (255, 255, 0)     # Cyan for skeleton lines
COLOR_TEXT = (255, 255, 255)       # White for text


def draw_landmarks(frame, pose_landmarks, frame_width, frame_height):
    """Draw landmarks and skeleton on frame."""
    landmark_positions = {}

    # Extract and draw landmarks
    for name, idx in LANDMARKS.items():
        lm = pose_landmarks[idx]
        px = int(lm.x * frame_width)
        py = int(lm.y * frame_height)
        landmark_positions[name] = (px, py, lm.visibility)

        # Choose color based on confidence
        color = COLOR_LANDMARK if lm.visibility >= MIN_LANDMARK_CONFIDENCE else COLOR_LOW_CONF

        # Draw landmark point
        cv2.circle(frame, (px, py), 6, color, -1)
        cv2.circle(frame, (px, py), 8, (0, 0, 0), 2)  # Black outline

    # Draw skeleton connections
    for start, end in SKELETON_CONNECTIONS:
        if start in landmark_positions and end in landmark_positions:
            p1 = landmark_positions[start][:2]
            p2 = landmark_positions[end][:2]
            cv2.line(frame, p1, p2, COLOR_SKELETON, 2)

    return frame


def visualize_video(input_path: str, output_path: str):
    """Process video and create annotated output."""
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {input_path}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps:.1f}, Frames: {total_frames}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Setup pose landmarker
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    frame_idx = 0
    detected_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect pose
            timestamp_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Draw landmarks if detected
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                frame = draw_landmarks(frame, result.pose_landmarks[0], frame_width, frame_height)
                detected_count += 1

            # Draw frame number
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

            out.write(frame)
            frame_idx += 1

            # Progress indicator
            if frame_idx % 30 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames...")

    finally:
        cap.release()
        out.release()
        landmarker.close()

    print(f"\nOutput: {output_path}")
    print(f"Pose detected in {detected_count}/{frame_idx} frames ({100*detected_count/frame_idx:.1f}%)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize pose detection on video")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("--output", "-o", help="Output video file path")
    args = parser.parse_args()

    # Generate default output path if not specified
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_annotated.mp4"

    visualize_video(args.input, output_path)


if __name__ == "__main__":
    main()
