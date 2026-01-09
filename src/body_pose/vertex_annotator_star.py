import cv2
import numpy as np
import supervision as sv
from supervision.annotators.base import ImageType
from supervision.keypoint.core import KeyPoints
from supervision.utils.conversion import ensure_cv2_image_for_annotation


class VertexAnnotatorStar(sv.VertexAnnotator):
    """
    Extend the original class to add vertices with the shape of heart or star.
    """

    def __init__(self, color=sv.Color.RED, radius=5, thickness=-1):
        super().__init__(color=color, radius=radius)
        self.thickness = thickness

    @ensure_cv2_image_for_annotation
    def annotate(self, scene: ImageType, key_points: KeyPoints) -> ImageType:
        """Draw a 5-pointed star on an image."""
        assert isinstance(scene, np.ndarray)
        if len(key_points) == 0:
            return scene

        for xy in key_points.xy:
            for x, y in xy:
                pts = np.array(
                    [
                        [x, y - self.radius],
                        [x + self.radius // 3, y - self.radius // 3],
                        [x + self.radius, y - self.radius // 3],
                        [x + self.radius // 2, y + self.radius // 6],
                        [x + 2 * self.radius // 3, y + self.radius],
                        [x, y + self.radius // 2],
                        [x - 2 * self.radius // 3, y + self.radius],
                        [x - self.radius // 2, y + self.radius // 6],
                        [x - self.radius, y - self.radius // 3],
                        [x - self.radius // 3, y - self.radius // 3],
                    ],
                    np.int32,
                ).reshape((-1, 1, 2))

                if self.thickness == -1:
                    cv2.fillPoly(scene, [pts], self.color.as_bgr())
                else:
                    cv2.polylines(
                        scene,
                        [pts],
                        isClosed=True,
                        color=self.color.as_bgr(),
                        thickness=self.thickness,
                    )

        return scene
