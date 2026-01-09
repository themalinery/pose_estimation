import torch
import supervision as sv
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
    infer_device,
)
from vit_pose.vertex_annotator_heart import VertexAnnotatorHeart


device = infer_device()

# # Detect humans in the image
person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained(
    "PekingU/rtdetr_r50vd_coco_o365", device_map=device
)

# Detect keypoints for each person found
image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
model = VitPoseForPoseEstimation.from_pretrained(
    "usyd-community/vitpose-base-simple", device_map=device
)


def vit_pose_estimation(image, frame_count: int):
    inputs = person_image_processor(images=image, return_tensors="pt").to(
        person_model.device
    )

    with torch.no_grad():
        outputs = person_model(**inputs)

    results = person_image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
    )
    result = results[0]

    # Human label refers 0 index in COCO dataset
    person_boxes = result["boxes"][result["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()

    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(
        model.device
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pose_results = image_processor.post_process_pose_estimation(
        outputs, boxes=[person_boxes]
    )
    image_pose_result = pose_results[0]

    xy = (
        torch.stack([pose_result["keypoints"] for pose_result in image_pose_result])
        .cpu()
        .numpy()
    )
    scores = (
        torch.stack([pose_result["scores"] for pose_result in image_pose_result])
        .cpu()
        .numpy()
    )

    color_edge_annotator = sv.Color.from_hex("#e1e1e1")

    color_vertex_annotator = sv.Color.from_hex("#ffc0cb")

    key_points = sv.KeyPoints(xy=xy, confidence=scores)

    edge_annotator = sv.EdgeAnnotator(color=color_edge_annotator, thickness=1)
    vertex_annotator = VertexAnnotatorHeart(color=color_vertex_annotator, radius=10)
    annotated_frame = edge_annotator.annotate(scene=image.copy(), key_points=key_points)
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame, key_points=key_points
    )

    return annotated_frame
