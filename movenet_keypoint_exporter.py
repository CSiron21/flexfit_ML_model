# --- SOCS | MoveNet Keypoint Exporter ---

import cv2
import numpy as np
import tensorflow as tf
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

# Ordered joint names by index for deterministic column naming
JOINT_NAMES_ORDERED = [name for name, _ in sorted(KEYPOINT_DICT.items(), key=lambda kv: kv[1])]

# --- Configurable Parameters (Edit these for your run) ---
VIDEO_FOLDER_PATH = "videos_dataset/squats/correct"     # Folder containing videos for this run
OUTPUT_DIR = "keypoints_data/squats/correct"            # Directory where outputs will be saved
METADATA_OUTPUT_DIR = "keypoints_data/squats/metadata"  # Directory where metadata JSON will be saved
CSV_FILENAME = "correct_squats.csv"              # CSV file name to save keypoints
LABEL_VALUE = 1.0                                       # 1.0 for correct, 0.0 for incorrect (set per run)
FRAME_SKIP = 3                                          # Process every nth frame
# Toggle intelligent cropping on/off
ENABLE_INTELLIGENT_CROPPING = False

# --- GPU Configuration ---
def setup_device():
    """
    Configure GPU if available, otherwise fall back to CPU.
    
    Returns:
        str: Device name ('GPU' or 'CPU')
    """
    print("🔍 Checking GPU availability...")
    
    # Check TensorFlow version and GPU support
    print(f"TensorFlow version: {tf.__version__}")
    
    try:
        # Check if CUDA is available (for NVIDIA GPUs)
        cuda_available = tf.test.is_built_with_cuda()
        print(f"CUDA support: {'✅ Available' if cuda_available else '❌ Not available'}")
        
        # Check for physical GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Physical GPU devices: {len(gpus)} found")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            
            # Try to enable memory growth for all GPUs
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ Memory growth enabled for all GPUs")
            except Exception as e:
                print(f"⚠️  Failed to enable memory growth: {e}")
            
            # Test if GPU is actually usable
            try:
                with tf.device('/GPU:0'):
                    # Simple test computation
                    test_tensor = tf.constant([1.0, 2.0, 3.0])
                    result = tf.reduce_sum(test_tensor)
                    print(f"✅ GPU test computation successful: {result.numpy()}")
                print("✅ GPU is functional and will be used")
                return 'GPU'
            except Exception as e:
                print(f"❌ GPU test failed: {e}")
                print("⚠️  Falling back to CPU")
                return 'CPU'
        else:
            print("❌ No physical GPU devices detected")
            
            # Check if it's an AMD GPU (TensorFlow doesn't support AMD GPUs well)
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ NVIDIA GPU detected via nvidia-smi")
                    print("⚠️  But TensorFlow can't access it - check CUDA/TensorFlow installation")
                else:
                    # Try to detect AMD GPU
                    try:
                        result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
                        if result.returncode == 0:
                            print("⚠️  AMD GPU detected - TensorFlow has limited AMD support")
                        else:
                            print("❌ No NVIDIA or AMD GPU detected")
                    except FileNotFoundError:
                        print("❌ No GPU detected")
            except FileNotFoundError:
                print("❌ nvidia-smi not found - no NVIDIA GPU or drivers")
            
            return 'CPU'
            
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        print("⚠️  Falling back to CPU")
        return 'CPU'

# Initialize device
DEVICE = setup_device()

# --- Load MoveNet Model ---
MODEL_PATH = "movenet-singlepose-thunder-4"

model = tf.saved_model.load(MODEL_PATH)
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

# --- MoveNet Inference Helper ---
def run_movenet_inference(input_image):
    """
    Run MoveNet inference on the selected device (GPU/CPU).
    
    Args:
        input_image: TensorFlow tensor of shape (1, height, width, 3)
    
    Returns:
        numpy.ndarray: Keypoints array of shape (1, 1, 17, 3)
    """
    with tf.device(DEVICE):
        keypoints = movenet(input_image)['output_0']
        return keypoints.numpy()

# --- Main Pose Detection Function ---
def detect_keypoints(image, crop_region=None):
    """Detects pose keypoints with smart cropping."""
    if not ENABLE_INTELLIGENT_CROPPING:
        # Always use full image padding; ignore crop region
        input_image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 256, 256)
        input_image = tf.cast(input_image, dtype=tf.int32)
        keypoints = run_movenet_inference(input_image)
        new_crop_region = None
        return keypoints[0][0], new_crop_region
    if crop_region is None:
        # First frame: use full image
        input_image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 256, 256)
    else:
        # Subsequent frames: use optimized crop
        input_image = crop_and_resize(image, crop_region)
    
    input_image = tf.cast(input_image, dtype=tf.int32)
    keypoints = run_movenet_inference(tf.expand_dims(input_image, axis=0))
    
    # Update crop region for next frame
    new_crop_region = determine_crop_region(keypoints, image.shape[0], image.shape[1])
    return keypoints[0][0], new_crop_region

def process_video(video_path, label, frame_skip=FRAME_SKIP):
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
            
            # Create data entry: label first, then descriptive keypoint columns
            # keypoints shape expected (17, 3) in [y, x, confidence] order
            row = {'label': float(label)}
            for j, name in enumerate(JOINT_NAMES_ORDERED):
                row[f'{name}_y'] = float(keypoints[j, 0])
                row[f'{name}_x'] = float(keypoints[j, 1])
                row[f'{name}_conf'] = float(keypoints[j, 2])
            keypoints_data.append(row)
        
        frame_num += 1
    
    cap.release()
    print(f"Extracted {len(keypoints_data)} frames from {os.path.basename(video_path)}")
    return keypoints_data

def process_dataset(dataset_path: str = VIDEO_FOLDER_PATH,
                    output_dir: str = OUTPUT_DIR,
                    metadata_output_dir: str = METADATA_OUTPUT_DIR,
                    csv_filename: str = CSV_FILENAME,
                    label: float = LABEL_VALUE,
                    frame_skip: int = FRAME_SKIP):
    """
    Process all videos in a single folder for one run.
    
    Args:
        dataset_path: Path to the videos folder (single class/run)
        output_dir: Directory to save outputs
        csv_filename: Name of the CSV file to write
        label: Label to assign to all frames from this run (1.0 or 0.0)
        frame_skip: Process every nth frame
    """
    print("=" * 60)
    print("MOVENET KEYPOINT EXTRACTION")
    print("=" * 60)
    print(f"🔧 Using device: {DEVICE}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_output_dir, exist_ok=True)
    
    all_keypoints_data = []
    video_stats = {}
    
    # Process videos in a single folder (one run = one label/class)
    if os.path.exists(dataset_path):
        print(f"\nProcessing videos from: {dataset_path}")
        for video_file in os.listdir(dataset_path):
            if video_file.lower().endswith((
                '.mp4', '.avi', '.mov', '.mkv', '.webm'
            )):
                video_path = os.path.join(dataset_path, video_file)
                video_data = process_video(video_path, label=label, frame_skip=frame_skip)
                all_keypoints_data.extend(video_data)
                video_stats[video_file] = {
                    'label': 'correct' if label == 1.0 else 'incorrect',
                    'frames_extracted': len(video_data)
                }
    
    if not all_keypoints_data:
        print("❌ No videos found or processed!")
        return None
    
    # Save keypoints data
    print(f"\nSaving keypoints data...")

    # Derive file paths
    csv_path = os.path.join(output_dir, csv_filename)
    stem = os.path.splitext(csv_filename)[0]
    metadata_path = os.path.join(metadata_output_dir, f"{stem}_metadata.json")

    # Save as CSV (label first, followed by <joint>_y, <joint>_x, <joint>_conf in index order)
    # Ensure column order is deterministic
    df = pd.DataFrame(all_keypoints_data)
    ordered_cols = ['label'] + [
        f'{name}_{comp}' for name in JOINT_NAMES_ORDERED for comp in ('y', 'x', 'conf')
    ]
    # Some legacy runs may lack certain cols; intersect to avoid KeyError
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    df = df[ordered_cols]
    df.to_csv(csv_path, index=False)

    # Save metadata
    metadata = {
        'total_frames': len(all_keypoints_data),
        'keypoint_structure': {
            'num_joints': 17,
            'features_per_joint': 3,
            'total_features': 51,
            'order': ['y', 'x', 'confidence']
        },
        'keypoint_names': JOINT_NAMES_ORDERED,
        'video_stats': video_stats,
        'processing_params': {
            'frame_skip': frame_skip,
            'min_confidence': MIN_CROP_KEYPOINT_SCORE
        },
        'label_mapping': {
            '1.0': 'correct',
            '0.0': 'incorrect'
        },
        'run_config': {
            'video_folder_path': dataset_path,
            'output_dir': output_dir,
            'csv_filename': csv_filename,
            'label_value': label
        }
    }

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
    print(f"   - {os.path.basename(csv_path)}")
    print(f"   - {os.path.basename(metadata_path)}")
    
    # Print video statistics
    print(f"\n📊 Video Statistics:")
    for video_name, stats in video_stats.items():
        print(f"   {video_name}: {stats['label']} ({stats['frames_extracted']} frames)")
    
    return all_keypoints_data

# --- Main execution ---
if __name__ == "__main__":
    # Process a single folder (configure paths at the top of this file)
    keypoints_data = process_dataset(
        dataset_path=VIDEO_FOLDER_PATH,
        output_dir=OUTPUT_DIR,
        metadata_output_dir=METADATA_OUTPUT_DIR,
        csv_filename=CSV_FILENAME,
        label=LABEL_VALUE,
        frame_skip=FRAME_SKIP,
    )

    if keypoints_data:
        print("\n🎉 Keypoint extraction completed successfully!")
        print("You can now use the extracted data for training your model.")
    else:
        print("\n❌ Keypoint extraction failed!")
        print("Please check your dataset structure and try again.")