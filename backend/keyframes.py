"""
Golf swing key frame detection.
Identifies address, top of backswing, and impact positions.
"""

from typing import Literal, TypedDict


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class KeyframeDetectionError(Exception):
    """Raised when key frames cannot be reliably identified."""
    pass


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

class HandPosition(TypedDict):
    x: float
    y: float
    source: Literal['wrist', 'shoulder']


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    # Address detection
    'stability_window': 5,              # frames for rolling std
    'stability_threshold': 0.015,       # max std for "stable"
    'movement_start_threshold': 0.03,   # velocity to detect swing start

    # Top of backswing
    'min_backswing_height_delta': 0.05, # min rise from address
    'min_backswing_ratio': 0.15,        # search starts at 15% of frames
    'max_backswing_ratio': 0.7,         # search ends at 70% of frames

    # Impact detection
    'min_impact_velocity': 0.08,        # min downward velocity at impact
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_hand_position(landmarks: dict) -> HandPosition:
    """
    Extract hand position from landmarks.
    Prefers wrist midpoint, falls back to shoulder midpoint.
    """
    if 'left_wrist' in landmarks and 'right_wrist' in landmarks:
        lw, rw = landmarks['left_wrist'], landmarks['right_wrist']
        return {
            'x': (lw['x'] + rw['x']) / 2,
            'y': (lw['y'] + rw['y']) / 2,
            'source': 'wrist'
        }
    elif 'left_wrist' in landmarks:
        lw = landmarks['left_wrist']
        return {'x': lw['x'], 'y': lw['y'], 'source': 'wrist'}
    elif 'right_wrist' in landmarks:
        rw = landmarks['right_wrist']
        return {'x': rw['x'], 'y': rw['y'], 'source': 'wrist'}
    else:
        ls, rs = landmarks['left_shoulder'], landmarks['right_shoulder']
        return {
            'x': (ls['x'] + rs['x']) / 2,
            'y': (ls['y'] + rs['y']) / 2,
            'source': 'shoulder'
        }


def _calculate_velocities(positions: list[float], dt: float) -> list[float]:
    """
    Calculate velocities using central difference method.
    Forward/backward difference at edges.
    """
    n = len(positions)
    if n < 2:
        return [0.0] * n

    velocities = []
    for i in range(n):
        if i == 0:
            v = (positions[1] - positions[0]) / dt
        elif i == n - 1:
            v = (positions[-1] - positions[-2]) / dt
        else:
            v = (positions[i + 1] - positions[i - 1]) / (2 * dt)
        velocities.append(v)

    return velocities


def _calculate_rolling_std(values: list[float], window: int) -> list[float]:
    """
    Calculate rolling standard deviation.
    Returns inf at edges where window doesn't fit.
    """
    n = len(values)
    result = []
    half = window // 2

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)

        if end - start < 2:
            result.append(float('inf'))
        else:
            w = values[start:end]
            mean = sum(w) / len(w)
            variance = sum((x - mean) ** 2 for x in w) / len(w)
            result.append(variance ** 0.5)

    return result


def _find_movement_start(velocities: list[float], threshold: float) -> int | None:
    """Find first frame where absolute velocity exceeds threshold."""
    for i, v in enumerate(velocities):
        if abs(v) > threshold:
            return i
    return None


# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def _detect_address(
    frames_data: list[dict],
    hand_positions: list[HandPosition],
    velocities: list[float],
    config: dict
) -> tuple[int, float]:
    """
    Detect address position: last stable frame before movement.
    Returns (frame_list_index, confidence).
    """
    y_positions = [hp['y'] for hp in hand_positions]
    rolling_std = _calculate_rolling_std(y_positions, config['stability_window'])

    movement_start = _find_movement_start(velocities, config['movement_start_threshold'])

    if movement_start is None:
        raise KeyframeDetectionError(
            "Could not detect swing start. Ensure video shows a complete swing."
        )

    # Search backwards from movement_start for stable frame
    address_idx = 0
    best_stability = float('inf')

    search_end = max(0, movement_start - 1)
    search_start = max(0, movement_start - config['stability_window'] * 3)

    for i in range(search_end, search_start - 1, -1):
        if rolling_std[i] < config['stability_threshold']:
            if rolling_std[i] < best_stability:
                best_stability = rolling_std[i]
                address_idx = i

    # Fallback: frame before movement
    if best_stability == float('inf'):
        address_idx = max(0, movement_start - 1)
        best_stability = rolling_std[address_idx]

    confidence = max(0.0, 1.0 - best_stability / config['stability_threshold'])
    return address_idx, confidence


def _detect_top_of_backswing(
    frames_data: list[dict],
    hand_positions: list[HandPosition],
    address_idx: int,
    config: dict
) -> tuple[int, float]:
    """
    Detect top of backswing: hands at highest point (min y).
    Returns (frame_list_index, confidence).
    """
    n = len(frames_data)

    min_idx = address_idx + int(n * config['min_backswing_ratio'])
    max_idx = address_idx + int(n * config['max_backswing_ratio'])
    max_idx = min(max_idx, n - 1)

    if min_idx >= max_idx:
        raise KeyframeDetectionError(
            "Video too short to identify backswing."
        )

    # Find minimum y (highest point)
    min_y = float('inf')
    top_idx = min_idx

    for i in range(min_idx, max_idx + 1):
        y = hand_positions[i]['y']
        if y < min_y:
            min_y = y
            top_idx = i

    # Validate height delta
    address_y = hand_positions[address_idx]['y']
    height_delta = address_y - min_y

    if height_delta < config['min_backswing_height_delta']:
        raise KeyframeDetectionError(
            "Could not detect backswing. Hands did not rise significantly."
        )

    confidence = min(1.0, height_delta / (config['min_backswing_height_delta'] * 2))
    return top_idx, confidence


def _detect_impact(
    frames_data: list[dict],
    hand_positions: list[HandPosition],
    velocities: list[float],
    top_idx: int,
    config: dict
) -> tuple[int, float, float]:
    """
    Detect impact: hands low with fast downward velocity.
    Returns (frame_list_index, confidence, velocity).
    """
    n = len(frames_data)

    if top_idx >= n - 1:
        raise KeyframeDetectionError(
            "Insufficient frames after backswing to detect impact."
        )

    # Find best impact frame: descent * velocity score
    best_score = -float('inf')
    impact_idx = top_idx + 1
    impact_velocity = 0.0

    top_y = hand_positions[top_idx]['y']

    for i in range(top_idx + 1, n):
        y = hand_positions[i]['y']
        v = velocities[i]

        # Only downward motion (positive v = moving down in frame)
        if v <= 0:
            continue

        descent = y - top_y
        score = descent * v

        if score > best_score:
            best_score = score
            impact_idx = i
            impact_velocity = v

    if impact_velocity < config['min_impact_velocity']:
        raise KeyframeDetectionError(
            "Could not detect impact. No frame with sufficient downward velocity."
        )

    confidence = min(1.0, impact_velocity / (config['min_impact_velocity'] * 2))
    return impact_idx, confidence, impact_velocity


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def detect_keyframes(analysis_result: dict, config: dict | None = None) -> dict:
    """
    Detect key frames in a golf swing video.

    Args:
        analysis_result: Output from analyzer.analyze_swing()
        config: Optional config overrides

    Returns:
        Dict with address, top_of_backswing, impact, detection_metadata

    Raises:
        KeyframeDetectionError: If key frames cannot be identified
    """
    config = {**DEFAULT_CONFIG, **(config or {})}

    frames_data = analysis_result['frames_data']
    video_info = analysis_result['video_info']

    if len(frames_data) < 10:
        raise KeyframeDetectionError(
            "Video too short for swing analysis. Need at least 10 valid frames."
        )

    # Time delta between processed frames
    dt = video_info['frame_skip'] / video_info['fps']

    # Extract hand positions and calculate velocities
    hand_positions = [_get_hand_position(f['landmarks']) for f in frames_data]
    y_positions = [hp['y'] for hp in hand_positions]
    velocities = _calculate_velocities(y_positions, dt)

    # Detect keyframes in sequence
    address_idx, address_conf = _detect_address(
        frames_data, hand_positions, velocities, config
    )
    top_idx, top_conf = _detect_top_of_backswing(
        frames_data, hand_positions, address_idx, config
    )
    impact_idx, impact_conf, impact_vel = _detect_impact(
        frames_data, hand_positions, velocities, top_idx, config
    )

    # Check if wrists were used
    used_wrist = any(hp['source'] == 'wrist' for hp in hand_positions)

    # Build result
    def build_keyframe(idx: int, conf: float, extra: dict | None = None) -> dict:
        frame = frames_data[idx]
        result = {
            'frame_index': frame['frame_index'],
            'timestamp': frame['frame_index'] / video_info['fps'],
            'landmarks': frame['landmarks'],
            'hand_position': hand_positions[idx],
            'confidence': conf,
        }
        if extra:
            result.update(extra)
        return result

    address_frame = frames_data[address_idx]
    impact_frame = frames_data[impact_idx]
    swing_duration = (impact_frame['frame_index'] - address_frame['frame_index']) / video_info['fps']

    return {
        'address': build_keyframe(address_idx, address_conf),
        'top_of_backswing': build_keyframe(top_idx, top_conf),
        'impact': build_keyframe(impact_idx, impact_conf, {'velocity': impact_vel}),
        'detection_metadata': {
            'total_frames_analyzed': len(frames_data),
            'swing_duration': swing_duration,
            'used_wrist_data': used_wrist,
        }
    }
