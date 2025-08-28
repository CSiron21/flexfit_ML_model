import numpy as np
import random
import csv
from typing import List, Tuple

# Configure variables
INPUT_CSV_PATH = "keypoints_data/overhead_presses/incorrect/incorrect_overhead_mirrored.csv"
OUTPUT_CSV_PATH = "keypoints_data/overhead_presses/incorrect/incorrect_overhead_jmirrored_jittered.csv"
MODE = "jitter"  # "jitter" or "mirror"
JITTER_RANGE = 0.03


# Keypoint dictionary mapping
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

# Minimum confidence threshold
MIN_CONFIDENCE = 0.2

def augment_with_jitter(keypoints, jitter_range=0.03):
    """
    Applies small random noise to (y, x) coordinates.
    
    Args:
        keypoints: Original keypoints array (17, 3)
        jitter_range: Maximum percentage jitter (e.g., 0.03 for ±3%)
        
    Returns:
        Augmented keypoints array (17, 3)
    """
    augmented = keypoints.copy()
    
    # Apply jitter to y and x coordinates for all keypoints
    for i in range(17):
        # Calculate jitter amount based on current coordinate value
        y_jitter = augmented[i, 0] * random.uniform(-jitter_range, jitter_range)
        x_jitter = augmented[i, 1] * random.uniform(-jitter_range, jitter_range)
        
        augmented[i, 0] += y_jitter  # y-coordinate
        augmented[i, 1] += x_jitter  # x-coordinate
    
    return augmented

def augment_with_mirroring(keypoints):
    """
    Flips the skeleton horizontally by mirroring x-coordinates.
    
    Args:
        keypoints: Original keypoints array (17, 3)
        
    Returns:
        Augmented keypoints array (17, 3)
    """
    augmented = keypoints.copy()
    
    # Mirror x-coordinates (1 - x)
    augmented[:, 1] = 1.0 - augmented[:, 1]
    
    # Swap left-right keypoints to maintain anatomical correctness
    left_right_pairs = [
        ('left_eye', 'right_eye'),
        ('left_ear', 'right_ear'),
        ('left_shoulder', 'right_shoulder'),
        ('left_elbow', 'right_elbow'),
        ('left_wrist', 'right_wrist'),
        ('left_hip', 'right_hip'),
        ('left_knee', 'right_knee'),
        ('left_ankle', 'right_ankle')
    ]
    
    for left, right in left_right_pairs:
        left_idx = KEYPOINT_DICT[left]
        right_idx = KEYPOINT_DICT[right]
        # Swap coordinates regardless of confidence
        augmented[[left_idx, right_idx], :2] = augmented[[right_idx, left_idx], :2]
    
    return augmented

def create_augmented_samples(keypoints, num_jitter=2, num_mirror=1):
    """
    Creates augmented samples from original keypoints.
    
    Args:
        keypoints: Original keypoints array (17, 3)
        num_jitter: Number of jitter-augmented samples to create
        num_mirror: Number of mirror-augmented samples to create
        
    Returns:
        List of augmented keypoints arrays
    """
    augmented_samples = []
    
    # Create jittered versions
    for _ in range(num_jitter):
        augmented_samples.append(augment_with_jitter(keypoints))
    
    # Create mirrored version
    for _ in range(num_mirror):
        augmented_samples.append(augment_with_mirroring(keypoints))
    
    return augmented_samples


def _row_to_label_and_keypoints(row: List[str]) -> Tuple[str, np.ndarray]:
    """
    Converts a CSV row to (label, keypoints[17,3]) assuming layout:
    label, y0, x0, c0, y1, x1, c1, ..., y16, x16, c16
    """
    label = row[0]
    values = list(map(float, row[1:]))
    if len(values) != 51:
        raise ValueError(f"Expected 51 keypoint values after label, got {len(values)}")
    keypoints = np.array(values, dtype=np.float32).reshape(17, 3)
    return label, keypoints


def _label_and_keypoints_to_row(label: str, keypoints: np.ndarray) -> List[str]:
    """
    Flattens (label, keypoints[17,3]) back to CSV row with 52 columns.
    """
    flat = keypoints.reshape(-1).tolist()
    return [label] + [f"{v:.6f}" for v in flat]


def process_csv(input_csv_path: str,
                output_csv_path: str,
                mode: str,
                jitter_range: float = 0.03) -> None:
    """
    Reads input CSV (label + 51 values), applies augmentation, writes augmented rows.

    Args:
        input_csv_path: Path to input CSV
        output_csv_path: Path to output CSV
        mode: 'jitter' or 'mirror'
        jitter_range: Max percentage jitter for jitter mode
    """
    if mode not in {"jitter", "mirror"}:
        raise ValueError("mode must be 'jitter' or 'mirror'")

    total_in = 0
    total_out = 0

    with open(input_csv_path, "r", newline="") as f_in, open(output_csv_path, "w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # Optional: try to detect header; if 52 columns and first not numeric, treat as header
        peek = next(reader, None)
        if peek is None:
            return

        def _looks_like_header(row: List[str]) -> bool:
            if len(row) != 52:
                return False
            try:
                # If all but first parse as float -> probably data
                _ = [float(x) for x in row[1:]]
                return False
            except Exception:
                return True

        has_header = _looks_like_header(peek)
        if has_header:
            # Write a header mirroring the input
            writer.writerow(peek)
        else:
            # Process the peek row as data
            reader = (r for r in [peek] + list(reader))

        for row in reader:
            if len(row) != 52:
                # Skip malformed rows
                continue
            total_in += 1
            try:
                label, keypoints = _row_to_label_and_keypoints(row)
            except Exception:
                continue

            if mode == "jitter":
                augmented = augment_with_jitter(keypoints, jitter_range=jitter_range)
                writer.writerow(_label_and_keypoints_to_row(label, augmented))
                total_out += 1
            else:  # mirror
                augmented = augment_with_mirroring(keypoints)
                writer.writerow(_label_and_keypoints_to_row(label, augmented))
                total_out += 1

    print(f"Processed {total_in} input rows -> wrote {total_out} augmented rows to '{output_csv_path}'.")

if __name__ == "__main__":

    process_csv(
        input_csv_path=INPUT_CSV_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        mode=MODE,
        jitter_range=JITTER_RANGE,
    )