# @title Default title text
try:
    from mediapipe.framework.formats import landmark_pb2
except ImportError:
    # Fallback for environments where mediapipe.framework is not available
    class _Landmark:
        """Proxy class that wraps a single landmark."""
        def __init__(self, x=0, y=0, z=0, visibility=1.0, presence=1.0):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = visibility
            self.presence = presence
        
        def HasField(self, field_name):
            return hasattr(self, field_name)
    
    class _NormalizedLandmarkList:
        """Proxy class that wraps mediapipe landmark data."""
        def __init__(self, landmarks=None):
            self.landmark = landmarks if landmarks else []
    
    class landmark_pb2:
        NormalizedLandmarkList = _NormalizedLandmarkList
        NormalizedLandmark = _Landmark

try:
    from mediapipe.python.solutions.drawing_utils import (
        DrawingSpec,
        _normalized_to_pixel_coordinates,
    )
except ImportError:
    # Fallback definitions
    from dataclasses import dataclass
    @dataclass
    class DrawingSpec:
        color: tuple = (0, 0, 255)
        thickness: int = 2
        circle_radius: int = 2
    
    def _normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
        if not (0 <= normalized_x <= 1 and 0 <= normalized_y <= 1):
            return None
        return int(normalized_x * image_width), int(normalized_y * image_height)

import cv2
from typing import List, Mapping, Optional, Tuple, Union
import numpy as np

PRESENCE_THRESHOLD = 0.5
RGB_CHANNELS = 3
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)
VISIBILITY_THRESHOLD = 0.5


def draw_landmarks(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[DrawingSpec, Mapping[int, DrawingSpec]] = DrawingSpec(
        color=RED_COLOR
    ),
    connection_drawing_spec: Union[
        DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]
    ] = DrawingSpec(),
):
    """Draws the landmarks and the connections on the image.

    Args:
      image: A three channel RGB image represented as numpy ndarray.
      landmark_list: A normalized landmark list proto message to be annotated on
        the image.
      connections: A list of landmark index tuples that specifies how landmarks to
        be connected in the drawing.
      landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
        drawing settings such as color, line thickness, and circle radius.
      connection_drawing_spec: A DrawingSpec object that specifies the
        connections' drawing settings such as color and line thickness.

    Raises:
      ValueError: If one of the followings:
        a) If the input image is not three channel RGB.
        b) If any connetions contain invalid landmark index.
    """
    if not landmark_list:
        return
    if image.shape[2] != RGB_CHANNELS:
        raise ValueError("Input image must contain three channel rgb data.")
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
            landmark.HasField("visibility")
            and landmark.visibility < VISIBILITY_THRESHOLD
        ) or (landmark.HasField("presence") and landmark.presence < PRESENCE_THRESHOLD):
            continue
        landmark_px = _normalized_to_pixel_coordinates(
            landmark.x, landmark.y, image_cols, image_rows
        )
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                if isinstance(connection_drawing_spec, Mapping):
                    cv2.line(
                        image,
                        idx_to_coordinates[start_idx],
                        idx_to_coordinates[end_idx],
                        connection_drawing_spec[connection].color,
                        connection_drawing_spec[connection].thickness,
                    )
                else:
                    cv2.line(
                        image,
                        idx_to_coordinates[start_idx],
                        idx_to_coordinates[end_idx],
                        connection_drawing_spec.color,
                        connection_drawing_spec.thickness,
                    )

    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    for idx, landmark_px in idx_to_coordinates.items():
        if isinstance(landmark_drawing_spec, Mapping):
            cv2.circle(
                img=image,
                center=(
                    int(landmark_px[0]) - landmark_drawing_spec[idx].circle_radius // 2,
                    int(landmark_px[1]),
                ),
                radius=landmark_drawing_spec[idx].circle_radius // 2,
                color=landmark_drawing_spec[idx].color,
                thickness=landmark_drawing_spec[idx].thickness,
            )
            cv2.circle(
                img=image,
                center=(
                    int(landmark_px[0]) + landmark_drawing_spec[idx].circle_radius // 2,
                    int(landmark_px[1]),
                ),
                radius=landmark_drawing_spec[idx].circle_radius // 2,
                color=landmark_drawing_spec[idx].color,
                thickness=landmark_drawing_spec[idx].thickness,
            )

            # Triangle (bottom of heart)
            pts = np.array(
                [
                    [
                        int(landmark_px[0]) - landmark_drawing_spec[idx].circle_radius,
                        int(landmark_px[1]),
                    ],
                    [
                        int(landmark_px[0]) + landmark_drawing_spec[idx].circle_radius,
                        int(landmark_px[1]),
                    ],
                    [
                        int(landmark_px[0]),
                        landmark_px[1] + landmark_drawing_spec[idx].circle_radius * 2,
                    ],
                ],
                np.int32,
            ).reshape((-1, 1, 2))

            cv2.fillPoly(image, [pts], landmark_drawing_spec[idx].color)

        else:
            # Two circles (top lobes of heart)
            cv2.circle(
                img=image,
                center=(
                    int(landmark_px[0]) - landmark_drawing_spec.circle_radius // 2,
                    int(landmark_px[1]),
                ),
                radius=landmark_drawing_spec.circle_radius // 2,
                color=landmark_drawing_spec.color,
                thickness=landmark_drawing_spec.thickness,
            )
            cv2.circle(
                img=image,
                center=(
                    int(landmark_px[0]) + landmark_drawing_spec.circle_radius // 2,
                    int(landmark_px[1]),
                ),
                radius=landmark_drawing_spec.circle_radius // 2,
                color=landmark_drawing_spec.color,
                thickness=landmark_drawing_spec.thickness,
            )

            # Triangle (bottom of heart)
            pts = np.array(
                [
                    [
                        int(landmark_px[0]) - landmark_drawing_spec.circle_radius,
                        int(landmark_px[1]),
                    ],
                    [
                        int(landmark_px[0]) + landmark_drawing_spec.circle_radius,
                        int(landmark_px[1]),
                    ],
                    [
                        int(landmark_px[0]),
                        landmark_px[1] + landmark_drawing_spec.circle_radius * 2,
                    ],
                ],
                np.int32,
            ).reshape((-1, 1, 2))

            cv2.fillPoly(image, [pts], landmark_drawing_spec.color)

            # cv2.circle(image, landmark_px, landmark_drawing_spec.circle_radius,
            # landmark_drawing_spec.color, landmark_drawing_spec.thickness)
