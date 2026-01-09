import mediapipe as mp
import os
from moviepy import ImageSequenceClip
from natsort import natsorted
from src.mediapipe_pose.utils import draw_landmarks
from src.mediapipe_pose.get_landmarks_and_connections import get_default_hand_connection_style, get_default_hand_landmark_style
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation, infer_device
from src.body_pose.vertex_annotator_heart import VertexAnnotatorHeart


def create_video_from_images(folder_path, output_video_file, fps):
    """
    Creates a video file from a sequence of images in a folder.

    Args:
        folder_path (str): The path to the folder containing the images.
        output_video_file (str): The name of the output video file (e.g., 'my_video.mp4').
        fps (int): The frames per second for the output video.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # List all image files in the folder.
    # We use natsorted to ensure files with numerical names (e.g., image-1.png, image-10.png)
    # are sorted in a human-friendly way.
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [
        os.path.join(folder_path, f)
        for f in natsorted(os.listdir(folder_path))
        if f.lower().endswith(supported_extensions)
    ]

    if not image_files:
        print(f"Error: No supported image files found in '{folder_path}'.")
        return

    if len(image_files) < 2:
        print("Error: At least two images are required to create a video.")
        return

    print(f"Found {len(image_files)} images. Creating video...")

    try:
        # Create a video clip from the list of image files.
        clip = ImageSequenceClip(image_files, fps=fps)

        # Write the video file to the specified path.
        clip.write_videofile(output_video_file, fps=fps)

        print(f"Successfully created video: '{output_video_file}'")
    except Exception as e:
        print(f"An error occurred while creating the video: {e}")

def process_hand_pose_estimation(path_video, output_folder):

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # landmark_annotations = mp_styles.get_default_hand_landmark_style()
    landmark_annotations = get_default_hand_landmark_style()
    connections_annotations = get_default_hand_connection_style()

    # Initialize video capture
    vidcap = cv2.VideoCapture(path_video)

    frame_count = 0
    annotated_frames = []
    frame_count = 0
    annotated_frames = []
    # Initialize hand tracking
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while vidcap.isOpened():
            ret, frame = vidcap.read()
            if not ret:
                break

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame for hand tracking
            processFrames = hands.process(rgb_frame)

            # Draw landmarks on the frame
            if processFrames.multi_hand_landmarks:
                for lm in processFrames.multi_hand_landmarks:
                    # mpdrawing.draw_landmarks(frame, lm, mphands.HAND_CONNECTIONS)
                    draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=landmark_annotations, connection_drawing_spec=connections_annotations)

            filename = f'{output_folder}/{frame_count}.jpg'
            cv2.imwrite(filename, frame)

            # Exit loop by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    # Release the video capture and close windows
    vidcap.release()
    cv2.destroyAllWindows()


def process_body_pose_estimation(path_video, output_folder):
    """
    Process video with body pose estimation, drawing keypoints and edges on each frame.
    
    Args:
        path_video (str): Path to input video file.
        output_folder (str): Folder to save annotated frames.
    """
    # Initialize device and models
    device = infer_device()
    
    # Detect humans in the image
    person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)
    
    # Detect keypoints for each person found
    image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)
    
    # Initialize video capture
    vidcap = cv2.VideoCapture(path_video)
    
    frame_count = 0
    
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret:
            break
        
        # Convert BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        
        # Detect humans in the image
        inputs = person_image_processor(images=image, return_tensors="pt").to(person_model.device)
        
        with torch.no_grad():
            outputs = person_model(**inputs)
        
        results = person_image_processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
        )
        result = results[0]
        
        # Human label refers to 0 index in COCO dataset
        person_boxes = result["boxes"][result["labels"] == 0]
        person_boxes = person_boxes.cpu().numpy()
        
        # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
        if len(person_boxes) > 0:
            person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
            person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
            
            inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
            image_pose_result = pose_results[0]
            
            xy = torch.stack([pose_result['keypoints'] for pose_result in image_pose_result]).cpu().numpy()
            scores = torch.stack([pose_result['scores'] for pose_result in image_pose_result]).cpu().numpy()
            
            # Set up annotators
            color_edge_annotator = sv.Color.from_hex("#e1e1e1")
            color_vertex_annotator = sv.Color.from_hex('#ffc0cb')
            
            key_points = sv.KeyPoints(
                xy=xy, confidence=scores
            )
            
            edge_annotator = sv.EdgeAnnotator(
                color=color_edge_annotator,
                thickness=1
            )
            vertex_annotator = VertexAnnotatorHeart(
                color=color_vertex_annotator,
                radius=10
            )
            
            annotated_frame = edge_annotator.annotate(
                scene=frame.copy(),
                key_points=key_points
            )
            annotated_frame = vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=key_points
            )
        else:
            annotated_frame = frame
        
        # Save frame
        filename = f'{output_folder}/{frame_count}.jpg'
        cv2.imwrite(filename, annotated_frame)
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Release the video capture
    vidcap.release()
    print(f"Body pose estimation complete. Total frames: {frame_count}")
