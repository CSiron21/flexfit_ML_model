# --- SOCS | MoveNet Keypoint Exporter ---

import cv2
import numpy as np
import tensorflow as tf
tf.__internal__.register_load_context_function = lambda x: None  # Monkey-Patch For Python 3.13 Compatibility
import tensorflow_hub as hub
import pandas as pd
import os
import json
from pathlib import Path

# --- Constants ---
MIN_CROP_KEYPOINT_SCORE = 0.2  # Min confidence to trust a keypoint
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# --- Load MoveNet Model ---
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures['serving_default']

# --- Crop Region Functions (from the second script) ---
def init_crop_region(image_height, image_width):
    """Default crop if no person is detected."""
    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width
    return {
        'y_min': y_min, 'x_min': x_min,
        'y_max': y_min + box_height, 'x_max': x_min + box_width,
        'height': box_height, 'width': box_width
    }

def torso_visible(keypoints):
    """Check if shoulders/hips are detected."""
    return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] > MIN_CROP_KEYPOINT_SCORE or
             keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] > MIN_CROP_KEYPOINT_SCORE) and
            (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] > MIN_CROP_KEYPOINT_SCORE or
             keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] > MIN_CROP_KEYPOINT_SCORE))

def determine_crop_region(keypoints, image_height, image_width):
    """Calculate optimal crop around the person."""
    target_keypoints = {
        joint: [keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
                keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width]
        for joint in KEYPOINT_DICT
    }

    if torso_visible(keypoints):
        center_y = (target_keypoints['left_hip'][0] + target_keypoints['right_hip'][0]) / 2
        center_x = (target_keypoints['left_hip'][1] + target_keypoints['right_hip'][1]) / 2

        # Estimate body bounds
        max_dist = max(
            abs(center_y - target_keypoints[joint][0]) * 1.2  # Padding
            for joint in KEYPOINT_DICT
            if keypoints[0, 0, KEYPOINT_DICT[joint], 2] > MIN_CROP_KEYPOINT_SCORE
        )
        crop_length_half = min(max_dist, center_x, center_y, image_width - center_x, image_height - center_y)

        return {
            'y_min': (center_y - crop_length_half) / image_height,
            'x_min': (center_x - crop_length_half) / image_width,
            'y_max': (center_y + crop_length_half) / image_height,
            'x_max': (center_x + crop_length_half) / image_width,
            'height': (2 * crop_length_half) / image_height,
            'width': (2 * crop_length_half) / image_width
        }
    else:
        return init_crop_region(image_height, image_width)

def crop_and_resize(image, crop_region, crop_size=(256, 256)):
    """Crop and resize the image for MoveNet."""
    boxes = [[crop_region['y_min'], crop_region['x_min'],
             crop_region['y_max'], crop_region['x_max']]]
    return tf.image.crop_and_resize(
        tf.expand_dims(image, axis=0), boxes, [0], crop_size
    )[0]

# --- Main Pose Detection Function ---
def detect_keypoints(image, crop_region=None):
    """Detects pose keypoints with smart cropping."""
    if crop_region is None:
        # First frame: use full image
        input_image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 256, 256)
    else:
        # Subsequent frames: use optimized crop
        input_image = crop_and_resize(image, crop_region)
    
    input_image = tf.cast(input_image, dtype=tf.int32)
    keypoints = movenet(tf.expand_dims(input_image, axis=0))['output_0'].numpy()
    
    # Update crop region for next frame
    new_crop_region = determine_crop_region(keypoints, image.shape[0], image.shape[1])
    return keypoints[0][0], new_crop_region

def process_video(video_path, label, frame_skip=3):
    """
    Process a single video and extract keypoints
    
    Args:
        video_path: Path to the video file
        label: Label for the video (1.0 for correct, 0.0 for incorrect)
        frame_skip: Process every nth frame
    
    Returns:
        List of dictionaries containing keypoint data
    """
    print(f"Processing: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    keypoints_data = []
    crop_region = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % frame_skip == 0:  # Process every nth frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keypoints, crop_region = detect_keypoints(rgb, crop_region)
            
            # Create data entry
            data_entry = {
                'video_name': os.path.basename(video_path),
                'frame_number': frame_num,
                'label': label,
                'keypoints': keypoints.flatten().tolist()  # Flatten to 1D array
            }
            keypoints_data.append(data_entry)
        
        frame_num += 1
    
    cap.release()
    print(f"  Extracted {len(keypoints_data)} frames from {os.path.basename(video_path)}")
    return keypoints_data

def process_dataset(dataset_path="dataset", output_dir="keypoints_data", frame_skip=3):
    """
    Process all videos in the dataset folder
    
    Args:
        dataset_path: Path to the dataset folder
        output_dir: Directory to save the extracted keypoints
        frame_skip: Process every nth frame
    """
    print("=" * 60)
    print("MOVENET KEYPOINT EXTRACTION")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_keypoints_data = []
    video_stats = {}
    
    # Process correct videos
    correct_path = os.path.join(dataset_path, "correct")
    if os.path.exists(correct_path):
        print(f"\nProcessing correct videos from: {correct_path}")
        for video_file in os.listdir(correct_path):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(correct_path, video_file)
                video_data = process_video(video_path, label=1.0, frame_skip=frame_skip)
                all_keypoints_data.extend(video_data)
                video_stats[video_file] = {
                    'label': 'correct',
                    'frames_extracted': len(video_data)
                }
    
    # Process incorrect videos
    incorrect_path = os.path.join(dataset_path, "incorrect")
    if os.path.exists(incorrect_path):
        print(f"\nProcessing incorrect videos from: {incorrect_path}")
        for video_file in os.listdir(incorrect_path):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(incorrect_path, video_file)
                video_data = process_video(video_path, label=0.0, frame_skip=frame_skip)
                all_keypoints_data.extend(video_data)
                video_stats[video_file] = {
                    'label': 'incorrect',
                    'frames_extracted': len(video_data)
                }
    
    if not all_keypoints_data:
        print("❌ No videos found or processed!")
        return None
    
    # Save keypoints data
    print(f"\nSaving keypoints data...")
    
    # Save as CSV
    df = pd.DataFrame(all_keypoints_data)
    csv_path = os.path.join(output_dir, "keypoints_dataset.csv")
    df.to_csv(csv_path, index=False)
    
    # Save as JSON for easier processing
    json_path = os.path.join(output_dir, "keypoints_dataset.json")
    with open(json_path, 'w') as f:
        json.dump(all_keypoints_data, f, indent=2)
    
    # Save metadata
    metadata = {
        'total_frames': len(all_keypoints_data),
        'keypoint_structure': {
            'num_joints': 17,
            'features_per_joint': 3,  # x, y, confidence
            'total_features': 51
        },
        'keypoint_names': list(KEYPOINT_DICT.keys()),
        'video_stats': video_stats,
        'processing_params': {
            'frame_skip': frame_skip,
            'min_confidence': MIN_CROP_KEYPOINT_SCORE
        },
        'label_mapping': {
            '1.0': 'correct',
            '0.0': 'incorrect'
        }
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETED!")
    print("=" * 60)
    print(f"✅ Total frames extracted: {len(all_keypoints_data)}")
    print(f"✅ Videos processed: {len(video_stats)}")
    print(f"✅ Output directory: {output_dir}")
    print(f"✅ Files created:")
    print(f"   - keypoints_dataset.csv")
    print(f"   - keypoints_dataset.json")
    print(f"   - metadata.json")
    
    # Print video statistics
    print(f"\n📊 Video Statistics:")
    for video_name, stats in video_stats.items():
        print(f"   {video_name}: {stats['label']} ({stats['frames_extracted']} frames)")
    
    return all_keypoints_data

def load_keypoints_data(data_path="keypoints_data"):
    """
    Load extracted keypoints data
    
    Args:
        data_path: Path to the keypoints data directory
    
    Returns:
        Dictionary containing the loaded data
    """
    csv_path = os.path.join(data_path, "keypoints_dataset.csv")
    metadata_path = os.path.join(data_path, "metadata.json")
    
    if not os.path.exists(csv_path):
        print(f"❌ Keypoints data not found at: {csv_path}")
        print("Please run the extraction first using process_dataset()")
        return None
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return {
        'dataframe': df,
        'metadata': metadata
    }

# --- Main execution ---
if __name__ == "__main__":
    # Process the entire dataset
    keypoints_data = process_dataset(
        dataset_path="dataset",
        output_dir="keypoints_data",
        frame_skip=3
    )
    
    if keypoints_data:
        print("\n🎉 Keypoint extraction completed successfully!")
        print("You can now use the extracted data for training your model.")
    else:
        print("\n❌ Keypoint extraction failed!")
        print("Please check your dataset structure and try again.")