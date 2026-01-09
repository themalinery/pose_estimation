
from typing import Mapping, Tuple
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import yaml
import os

# Load hand drawing config from config.yaml
def load_hand_drawing_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    hand_cfg = config.get('hand_drawing', {})
    radius = hand_cfg.get('radius', 20)
    color_landmarks = tuple(hand_cfg.get('color_landmarks', [179, 124, 247]))
    color_connections = tuple(hand_cfg.get('color_connections', [225, 225, 225]))
    return radius, color_landmarks, color_connections

_RADIUS, COLOR_LANDMARKS, COLOR_CONNECTIONS = load_hand_drawing_config()

# Hands
_THICKNESS_WRIST_MCP = 5
_THICKNESS_FINGER = 5
_THICKNESS_DOT = -1

# Hand landmarks
_PALM_LANMARKS = (HandLandmark.WRIST, HandLandmark.THUMB_CMC,
                  HandLandmark.INDEX_FINGER_MCP, HandLandmark.MIDDLE_FINGER_MCP,
                  HandLandmark.RING_FINGER_MCP, HandLandmark.PINKY_MCP)
_THUMP_LANDMARKS = (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP,
                    HandLandmark.THUMB_TIP)
_INDEX_FINGER_LANDMARKS = (HandLandmark.INDEX_FINGER_PIP,
                           HandLandmark.INDEX_FINGER_DIP,
                           HandLandmark.INDEX_FINGER_TIP)
_MIDDLE_FINGER_LANDMARKS = (HandLandmark.MIDDLE_FINGER_PIP,
                            HandLandmark.MIDDLE_FINGER_DIP,
                            HandLandmark.MIDDLE_FINGER_TIP)
_RING_FINGER_LANDMARKS = (HandLandmark.RING_FINGER_PIP,
                          HandLandmark.RING_FINGER_DIP,
                          HandLandmark.RING_FINGER_TIP)
_PINKY_FINGER_LANDMARKS = (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP,
                           HandLandmark.PINKY_TIP)

_HAND_LANDMARK_STYLE = {
    _PALM_LANMARKS:
        DrawingSpec(
            color=COLOR_LANDMARKS, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _THUMP_LANDMARKS:
        DrawingSpec(
            color=COLOR_LANDMARKS, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _INDEX_FINGER_LANDMARKS:
        DrawingSpec(
            color=COLOR_LANDMARKS, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _MIDDLE_FINGER_LANDMARKS:
        DrawingSpec(
            color=COLOR_LANDMARKS, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _RING_FINGER_LANDMARKS:
        DrawingSpec(
            color=COLOR_LANDMARKS, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
    _PINKY_FINGER_LANDMARKS:
        DrawingSpec(
            color=COLOR_LANDMARKS, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
}

# Hand connections
_PALM_CONNECTIONS = ((HandLandmark.WRIST, HandLandmark.THUMB_CMC),
                     (HandLandmark.WRIST, HandLandmark.INDEX_FINGER_MCP),
                     (HandLandmark.MIDDLE_FINGER_MCP,
                      HandLandmark.RING_FINGER_MCP),
                     (HandLandmark.RING_FINGER_MCP, HandLandmark.PINKY_MCP),
                     (HandLandmark.INDEX_FINGER_MCP,
                      HandLandmark.MIDDLE_FINGER_MCP), (HandLandmark.WRIST,
                                                        HandLandmark.PINKY_MCP))
_THUMB_CONNECTIONS = ((HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP),
                      (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
                      (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP))
_INDEX_FINGER_CONNECTIONS = ((HandLandmark.INDEX_FINGER_MCP,
                              HandLandmark.INDEX_FINGER_PIP),
                             (HandLandmark.INDEX_FINGER_PIP,
                              HandLandmark.INDEX_FINGER_DIP),
                             (HandLandmark.INDEX_FINGER_DIP,
                              HandLandmark.INDEX_FINGER_TIP))
_MIDDLE_FINGER_CONNECTIONS = ((HandLandmark.MIDDLE_FINGER_MCP,
                               HandLandmark.MIDDLE_FINGER_PIP),
                              (HandLandmark.MIDDLE_FINGER_PIP,
                               HandLandmark.MIDDLE_FINGER_DIP),
                              (HandLandmark.MIDDLE_FINGER_DIP,
                               HandLandmark.MIDDLE_FINGER_TIP))
_RING_FINGER_CONNECTIONS = ((HandLandmark.RING_FINGER_MCP,
                             HandLandmark.RING_FINGER_PIP),
                            (HandLandmark.RING_FINGER_PIP,
                             HandLandmark.RING_FINGER_DIP),
                            (HandLandmark.RING_FINGER_DIP,
                             HandLandmark.RING_FINGER_TIP))
_PINKY_FINGER_CONNECTIONS = ((HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
                             (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
                             (HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP))

_HAND_CONNECTION_STYLE = {
    _PALM_CONNECTIONS:
        DrawingSpec(color=COLOR_CONNECTIONS, thickness=_THICKNESS_WRIST_MCP),
    _THUMB_CONNECTIONS:
        DrawingSpec(color=COLOR_CONNECTIONS, thickness=_THICKNESS_FINGER),
    _INDEX_FINGER_CONNECTIONS:
        DrawingSpec(color=COLOR_CONNECTIONS, thickness=_THICKNESS_FINGER),
    _MIDDLE_FINGER_CONNECTIONS:
        DrawingSpec(color=COLOR_CONNECTIONS, thickness=_THICKNESS_FINGER),
    _RING_FINGER_CONNECTIONS:
        DrawingSpec(color=COLOR_CONNECTIONS, thickness=_THICKNESS_FINGER),
    _PINKY_FINGER_CONNECTIONS:
        DrawingSpec(color=COLOR_CONNECTIONS, thickness=_THICKNESS_FINGER)
}


def get_default_hand_landmark_style() -> Mapping[int, DrawingSpec]:
    """Returns the default hand landmark drawing style.

    Returns:
        A mapping from each hand landmark to the default drawing spec.
    """
    hand_landmark_style = {}
    for k, v in _HAND_LANDMARK_STYLE.items():
        for landmark in k:
            hand_landmark_style[landmark] = v
    return hand_landmark_style


def get_default_hand_connection_style(
) -> Mapping[Tuple[int, int], DrawingSpec]:
    """Returns the default hand connection drawing style.

    Returns:
        A mapping from each hand connection to the default drawing spec.
    """
    hand_connection_style = {}
    for k, v in _HAND_CONNECTION_STYLE.items():
        for connection in k:
            hand_connection_style[connection] = v
    return hand_connection_style