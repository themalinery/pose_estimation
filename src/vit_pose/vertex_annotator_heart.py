import cv2
import numpy as np
import supervision as sv
from supervision.annotators.base import ImageType
from supervision.keypoint.core import KeyPoints
from supervision.utils.conversion import ensure_cv2_image_for_annotation


class VertexAnnotatorHeart(sv.VertexAnnotator):
    """
    Extend the original class to add vertices with the shape of heart or star.
    """

    def __init__(self, color=sv.Color.RED, radius=5):
        super().__init__(color=color, radius=radius)

    @ensure_cv2_image_for_annotation
    def annotate(self, scene: ImageType, key_points: KeyPoints) -> ImageType:
        assert isinstance(scene, np.ndarray)
        if len(key_points) == 0:
            return scene

        for xy in key_points.xy:
            for x, y in xy:
                # Two circles (top lobes of heart)
                cv2.circle(
                    img=scene,
                    center=(int(x) - self.radius // 2, int(y)),
                    radius=self.radius // 2,
                    color=self.color.as_bgr(),
                    thickness=-1,
                )
                cv2.circle(
                    img=scene,
                    center=(int(x) + self.radius // 2, int(y)),
                    radius=self.radius // 2,
                    color=self.color.as_bgr(),
                    thickness=-1,
                )

                # Triangle (bottom of heart)
                pts = np.array(
                    [
                        [int(x) - self.radius, int(y)],
                        [int(x) + self.radius, int(y)],
                        [int(x), y + self.radius * 2],
                    ],
                    np.int32,
                ).reshape((-1, 1, 2))

                cv2.fillPoly(scene, [pts], self.color.as_bgr())

        return scene
