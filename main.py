import yaml
import shutil
from datetime import datetime
from pathlib import Path
from src.utils import (
    create_video_from_images,
    process_hand_pose_estimation,
    process_body_pose_estimation,
)


def get_paths_from_config(config: dict) -> tuple[Path, Path, Path]:
    """Extract paths from configuration dictionary."""

    input_path = Path(config["input_path"])
    output_dir = Path(config["output_dir"])
    output_name = config.get("output_name")
    task = config.get("task")
    frames_dir = Path(config.get("frames_dir"))

    output_dir = output_dir.joinpath(task)
    output_dir.mkdir(parents=True, exist_ok=True)

    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = frames_dir.joinpath(date)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if output_name:
        output_path = output_dir.joinpath(output_name)
    else:
        output_path = output_dir.joinpath(input_path.name)

    return input_path, output_path, frames_dir


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    input_path, output_path, frames_dir = get_paths_from_config(config)

    task = config.get("task")

    if task == "hand_pose_estimation":
        print("HAND POSE ESTIMATION SELECTED")
        process_hand_pose_estimation(str(input_path), str(frames_dir), config['drawing_settings'])
        create_video_from_images(str(frames_dir), str(output_path), fps=30)
        shutil.rmtree(frames_dir)
    elif task == "body_pose_estimation":
        print("BODY POSE ESTIMATION SELECTED")
        process_body_pose_estimation(str(input_path), str(frames_dir), config['drawing_settings'])
        create_video_from_images(str(frames_dir), str(output_path), fps=30)
        shutil.rmtree(frames_dir)
    else:
        print(f"Unknown task: {task}")


main()
