import gradio as gr
import shutil
from datetime import datetime
from pathlib import Path
import tempfile
import yaml
from src.utils import (
    create_video_from_images,
    process_hand_pose_estimation,
    process_body_pose_estimation,
)
from src.mediapipe_pose.get_landmarks_and_connections import (
    get_default_hand_landmark_style,
    get_default_hand_connection_style,
)


def hex_to_bgr(color_input):
    """Convert color to BGR tuple. Handles hex strings, rgb() strings, and tuples."""
    try:
        # If it's already a tuple, assume it's RGB and convert to BGR
        if isinstance(color_input, (tuple, list)) and len(color_input) == 3:
            r, g, b = color_input
            return (b, g, r)
        
        # If it's a string, try different formats
        if isinstance(color_input, str):
            color_input = color_input.strip()
            
            # Handle rgb(r, g, b) format
            if color_input.startswith('rgb'):
                color_input = color_input.replace('rgb(', '').replace(')', '').strip()
                parts = [int(x.strip()) for x in color_input.split(',')]
                if len(parts) == 3:
                    r, g, b = parts
                    return (b, g, r)
            
            # Handle hex format #RRGGBB
            if color_input.startswith('#'):
                hex_color = color_input.lstrip('#')
                if len(hex_color) == 6:
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    return (b, g, r)
        
        # Default fallback
        return (179, 124, 247)  # Default landmark purple in BGR
    
    except Exception as e:
        print(f"Warning: Color conversion failed for {color_input}: {e}")
        return (179, 124, 247)  # Default fallback


def process_video(
    video_file,
    task_type,
    landmark_radius,
    landmark_color,
    connection_color,
    connection_thickness,
):
    """
    Process video with pose estimation and return output video path.
    
    Args:
        video_file: Uploaded video file
        task_type: "hand_pose_estimation" or "body_pose_estimation"
        landmark_radius: Radius size for landmarks (int)
        landmark_color: Color for landmarks (hex string or tuple)
        connection_color: Color for connections (hex string or tuple)
        connection_thickness: Thickness for connections (int)
    
    Returns:
        Path to output video file
    """
    if video_file is None:
        raise gr.Error("Please upload a video file")
    
    try:
        # Create temporary directories
        temp_dir = Path(tempfile.gettempdir()) / f"pose_est_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        output_dir = temp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = output_dir / "output.mp4"
        
        # Convert colors to BGR tuples
        landmark_color_tuple = hex_to_bgr(landmark_color)
        connection_color_tuple = hex_to_bgr(connection_color)
        
        # Create drawing settings dictionary from interface input
        drawing_settings = {
            "color_landmarks": landmark_color_tuple,
            "color_connections": connection_color_tuple,
            "radius": int(landmark_radius),
            "thickness": int(connection_thickness)
        }
        
        print(f"Drawing settings: {drawing_settings}")
        
        # Process video based on task type
        if task_type == "hand_pose_estimation":
            process_hand_pose_estimation(
                video_file, 
                str(frames_dir),
                drawing_settings
            )
        elif task_type == "body_pose_estimation":
            process_body_pose_estimation(
                video_file, 
                str(frames_dir),
                drawing_settings
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Create video from processed frames
        create_video_from_images(str(frames_dir), str(output_video_path), fps=30)
        
        # Cleanup frames
        shutil.rmtree(frames_dir)
        
        return str(output_video_path)
    
    except Exception as e:
        raise gr.Error(f"Error processing video: {str(e)}")


# Create Gradio interface
with gr.Blocks(title="Pose Estimation") as demo:
    gr.Markdown("# Pose Estimation")
    gr.Markdown("Upload a video and process it with hand or body pose estimation")
    
    with gr.Row():
        with gr.Column():
            # Input controls
            gr.Markdown("## Input Settings")
            
            task_dropdown = gr.Radio(
                choices=["hand_pose_estimation", "body_pose_estimation"],
                value="hand_pose_estimation",
                label="Select Pose Estimation Task"
            )
            
            landmark_radius = gr.Number(
                value=20,
                label="Landmark Radius",
                minimum=1,
                maximum=100,
                step=1,
                precision=0
            )
            
            landmark_color = gr.Textbox(
                value="#B37CF7",  # Default: RGB(179, 124, 247) purple
                label="Landmark Color (hex, e.g. #FF0000 for red)",
                placeholder="#B37CF7"
            )
            
            connection_color = gr.Textbox(
                value="#E1E1E1",  # Default: RGB(225, 225, 225) gray
                label="Connection Color (hex, e.g. #00FF00 for green)",
                placeholder="#E1E1E1"
            )
            
            connection_thickness = gr.Number(
                value=5,
                label="Connection Thickness",
                minimum=1,
                maximum=20,
                step=1,
                precision=0
            )
            
            video_upload = gr.File(
                label="Upload Video",
                file_types=["video"],
                file_count="single"
            )
            
            process_button = gr.Button(
                "Process Video",
                variant="primary",
                size="lg"
            )
        
        with gr.Column():
            # Output section
            gr.Markdown("## Output")
            
            video_output = gr.Video(
                label="Processed Video",
                interactive=False
            )
            
            download_button = gr.File(
                label="Download Processed Video",
                visible=False
            )
    
    # Progress indicator
    progress_text = gr.Textbox(
        label="Status",
        interactive=False,
        visible=False
    )
    
    # Handle processing
    def process_and_update(video, task, radius, land_color, conn_color, conn_thickness):
        try:
            # Update status
            gr.Info("Processing video... This may take a few minutes.")
            
            output_path = process_video(
                video,
                task,
                int(radius),
                land_color,
                conn_color,
                int(conn_thickness)
            )
            
            gr.Info("Video processing complete!")
            
            return output_path, output_path
        except Exception as e:
            raise gr.Error(f"Processing failed: {str(e)}")
    
    process_button.click(
        fn=process_and_update,
        inputs=[video_upload, task_dropdown, landmark_radius, landmark_color, connection_color, connection_thickness],
        outputs=[video_output, download_button]
    )


if __name__ == "__main__":
    demo.launch()
