import numpy as np

# 21 landmarks x (x, y, z) = 63 features. Shared by data collection,
# training-time expectations, and live prediction so they can NEVER drift apart.
EXPECTED_FEATURES = 63


def extract_normalized_landmarks(hand_landmarks):
    """Turn a MediaPipe hand into 63 position- and scale-invariant features.

    Steps:
      1. Pull 21 (x, y, z) points.
      2. Move the origin to the wrist (landmark 0) so hand POSITION doesn't matter.
      3. Divide by palm size (wrist -> middle-finger MCP, landmark 9) so hand
         SIZE / camera distance doesn't matter.
      4. Flatten to a flat list of 63 floats.

    Returns a list of 63 floats, or None if no valid hand.

    This is the ONLY place this logic should live. collect_asl_data.py and
    predict_asl_live.py both import it, which guarantees the features used for
    training and for live prediction are identical.
    """
    if hand_landmarks is None:
        return None

    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    if pts.shape[0] != 21:
        return None

    # 2) wrist-relative
    pts = pts - pts[0]

    # 3) scale by palm size (distance wrist -> landmark 9)
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        # fallback: largest distance from wrist to any landmark
        scale = np.max(np.linalg.norm(pts, axis=1))
    if scale < 1e-6:
        return None  # degenerate / all points stacked

    pts = pts / scale

    # 4) flatten -> 63 floats
    return pts.flatten().tolist()