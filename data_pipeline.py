import os
import pandas as pd
import numpy as np
from movenet_keypoint_exporter import process_dataset, load_keypoints_data
import cv2
from sklearn.model_selection import train_test_split
import json

class FlexFitDataPipeline:
    """
    Connects MoveNet keypoint extraction with model training data preparation
    """
    
    def __init__(self, dataset_path="dataset", keypoints_dir="keypoints_data"):
        self.dataset_path = dataset_path
        self.keypoints_dir = keypoints_dir
        self.correct_videos_path = os.path.join(dataset_path, "correct")
        self.incorrect_videos_path = os.path.join(dataset_path, "incorrect")
        
        # Keypoint mapping for consistency
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def extract_keypoints_from_videos(self, frame_skip=3):
        """
        Extract keypoints from all videos using the updated MoveNet exporter
        """
        print("Extracting keypoints from videos...")
        
        # Use the updated keypoint exporter
        keypoints_data = process_dataset(
            dataset_path=self.dataset_path,
            output_dir=self.keypoints_dir,
            frame_skip=frame_skip
        )
        
        return keypoints_data
    
    def load_existing_keypoints(self):
        """
        Load existing keypoints data if available
        """
        print("Loading existing keypoints data...")
        
        loaded_data = load_keypoints_data(self.keypoints_dir)
        if loaded_data is None:
            return None
        
        return loaded_data['dataframe'], loaded_data['metadata']
    
    def prepare_training_data(self, df, test_size=0.2, val_size=0.2):
        """
        Prepare training data from keypoints dataframe
        """
        print(f"Preparing training data from {len(df)} samples...")
        
        # Convert keypoints string back to numpy array
        keypoints_list = []
        for keypoints_str in df['keypoints']:
            if isinstance(keypoints_str, str):
                # Parse string representation of list
                keypoints = eval(keypoints_str)
            else:
                keypoints = keypoints_str
            keypoints_list.append(np.array(keypoints).reshape(17, 3))
        
        X = np.array(keypoints_list)
        y_form = np.array(df['label'])
        
        # Create joint alignment labels (simplified - can be enhanced)
        # For now, assume joints are aligned if confidence > 0.5
        y_alignment = (X[:, :, 2] > 0.5).astype(np.float32)  # confidence scores
        
        # Create joint angles (simplified calculation)
        y_angles = self.calculate_joint_angles(X)
        
        # Split data
        X_temp, X_test, y_form_temp, y_form_test, y_align_temp, y_align_test, y_angles_temp, y_angles_test = train_test_split(
            X, y_form, y_alignment, y_angles, test_size=test_size, random_state=42, stratify=y_form
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_form_train, y_form_val, y_align_train, y_align_val, y_angles_train, y_angles_val = train_test_split(
            X_temp, y_form_temp, y_align_temp, y_angles_temp, test_size=val_ratio, random_state=42, stratify=y_form_temp
        )
        
        # Create output dictionaries
        train_data = {
            'X': X_train,
            'y_form': y_form_train.reshape(-1, 1),
            'y_alignment': y_align_train,
            'y_angles': y_angles_train
        }
        
        val_data = {
            'X': X_val,
            'y_form': y_form_val.reshape(-1, 1),
            'y_alignment': y_align_val,
            'y_angles': y_angles_val
        }
        
        test_data = {
            'X': X_test,
            'y_form': y_form_test.reshape(-1, 1),
            'y_alignment': y_align_test,
            'y_angles': y_angles_test
        }
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return train_data, val_data, test_data
    
    def calculate_joint_angles(self, keypoints):
        """
        Calculate joint angles from keypoints (simplified)
        """
        batch_size = keypoints.shape[0]
        angles = np.zeros((batch_size, 17))
        
        for i in range(batch_size):
            kp = keypoints[i]
            
            # Calculate angles for major joints (simplified)
            # Shoulder angles
            if kp[5, 2] > 0.3 and kp[6, 2] > 0.3 and kp[7, 2] > 0.3:  # left shoulder, right shoulder, left elbow
                left_shoulder_angle = self.calculate_angle(kp[5], kp[6], kp[7])
                angles[i, 5] = left_shoulder_angle
            
            if kp[6, 2] > 0.3 and kp[5, 2] > 0.3 and kp[8, 2] > 0.3:  # right shoulder, left shoulder, right elbow
                right_shoulder_angle = self.calculate_angle(kp[6], kp[5], kp[8])
                angles[i, 6] = right_shoulder_angle
            
            # Elbow angles
            if kp[5, 2] > 0.3 and kp[7, 2] > 0.3 and kp[9, 2] > 0.3:  # left shoulder, left elbow, left wrist
                left_elbow_angle = self.calculate_angle(kp[5], kp[7], kp[9])
                angles[i, 7] = left_elbow_angle
            
            if kp[6, 2] > 0.3 and kp[8, 2] > 0.3 and kp[10, 2] > 0.3:  # right shoulder, right elbow, right wrist
                right_elbow_angle = self.calculate_angle(kp[6], kp[8], kp[10])
                angles[i, 8] = right_elbow_angle
        
        return angles
    
    def calculate_angle(self, p1, p2, p3):
        """
        Calculate angle between three points
        """
        # Convert to 2D coordinates (x, y)
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def save_datasets(self, train_data, val_data, test_data, output_dir="processed_data"):
        """
        Save processed datasets
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as numpy arrays
        np.save(os.path.join(output_dir, "X_train.npy"), train_data['X'])
        np.save(os.path.join(output_dir, "y_form_train.npy"), train_data['y_form'])
        np.save(os.path.join(output_dir, "y_alignment_train.npy"), train_data['y_alignment'])
        np.save(os.path.join(output_dir, "y_angles_train.npy"), train_data['y_angles'])
        
        np.save(os.path.join(output_dir, "X_val.npy"), val_data['X'])
        np.save(os.path.join(output_dir, "y_form_val.npy"), val_data['y_form'])
        np.save(os.path.join(output_dir, "y_alignment_val.npy"), val_data['y_alignment'])
        np.save(os.path.join(output_dir, "y_angles_val.npy"), val_data['y_angles'])
        
        np.save(os.path.join(output_dir, "X_test.npy"), test_data['X'])
        np.save(os.path.join(output_dir, "y_form_test.npy"), test_data['y_form'])
        np.save(os.path.join(output_dir, "y_alignment_test.npy"), test_data['y_alignment'])
        np.save(os.path.join(output_dir, "y_angles_test.npy"), test_data['y_angles'])
        
        # Save metadata
        metadata = {
            'keypoint_names': self.keypoint_names,
            'feature_names': ['x', 'y', 'confidence'],
            'num_joints': 17,
            'num_features': 3,
            'train_samples': train_data['X'].shape[0],
            'val_samples': val_data['X'].shape[0],
            'test_samples': test_data['X'].shape[0],
            'label_mapping': {
                'form_score': {'0.0': 'incorrect', '1.0': 'correct'},
                'joint_alignment': 'confidence > 0.5',
                'joint_angles': 'degrees'
            }
        }
        
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Datasets saved to: {output_dir}")
        return output_dir

def run_data_pipeline():
    """
    Run the complete data pipeline
    """
    print("=" * 50)
    print("FLEXFIT DATA PIPELINE")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = FlexFitDataPipeline()
    
    # Check if keypoints data already exists
    if os.path.exists(os.path.join(pipeline.keypoints_dir, "keypoints_dataset.csv")):
        print("Found existing keypoints data. Loading...")
        df, metadata = pipeline.load_existing_keypoints()
        if df is None:
            print("Failed to load existing data. Extracting keypoints...")
            pipeline.extract_keypoints_from_videos()
            df, metadata = pipeline.load_existing_keypoints()
    else:
        print("No existing keypoints data found. Extracting keypoints...")
        pipeline.extract_keypoints_from_videos()
        df, metadata = pipeline.load_existing_keypoints()
    
    if df is None:
        print("❌ Failed to load keypoints data!")
        return None
    
    # Create training datasets
    print("\n2. Creating training datasets...")
    train_data, val_data, test_data = pipeline.prepare_training_data(df)
    
    # Save datasets
    print("\n3. Saving processed datasets...")
    output_dir = pipeline.save_datasets(train_data, val_data, test_data)
    
    print("\n" + "=" * 50)
    print("DATA PIPELINE COMPLETED!")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print("Files created:")
    print("  - X_train.npy, y_form_train.npy, y_alignment_train.npy, y_angles_train.npy")
    print("  - X_val.npy, y_form_val.npy, y_alignment_val.npy, y_angles_val.npy")
    print("  - X_test.npy, y_form_test.npy, y_alignment_test.npy, y_angles_test.npy")
    print("  - metadata.json")
    
    return train_data, val_data, test_data

if __name__ == "__main__":
    run_data_pipeline()
