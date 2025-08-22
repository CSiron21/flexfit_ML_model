"""
FlexFit ML Model Training Script

This script trains pose analysis CNN models with configurable architecture and data sources.
Supports dynamic model import from the cnns/ folder and comprehensive training pipeline.

STANDARDIZED CNN ARCHITECTURE:
All CNNs must implement the standardized 3-output structure:
{
    "form_score": (batch, 1) - Trainable output for form assessment
    "instruction_id": (batch,) - Rule-based instruction output (inference only)
    "joint_masked_keypoints": (batch, 17, 3) - Rule-based joint masking (inference only)
}

TRAINING FOCUS:
- Only form_score is trained using binary_crossentropy loss and accuracy metric
- instruction_id and joint_masked_keypoints are preserved for inference
- No modifications to internal CNN architecture or engineered features
"""

import os
import sys
import importlib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

# =============================================================================
# CONFIGURABLE PARAMETERS - Modify these as needed
# =============================================================================

# CNN Architecture to import dynamically from cnns/ folder
CNN_MODULE = "cnns.squat_pose_cnn"  # e.g., "cnns.squat_pose_cnn", "cnns.base_pose_cnn"

# Data file paths (per class: original, mirrored, original_jittered, mirrored_jittered)
CORRECT_ORIG_CSV_PATH = "keypoints_data/correct_original.csv"
CORRECT_MIRROR_CSV_PATH = "keypoints_data/correct_mirrored.csv"
CORRECT_JITTER_CSV_PATH = "keypoints_data/correct_jittered.csv"
CORRECT_MIRROR_JITTER_CSV_PATH = "keypoints_data/correct_mirrored_jittered.csv"

INCORRECT_ORIG_CSV_PATH = "keypoints_data/incorrect_original.csv"
INCORRECT_MIRROR_CSV_PATH = "keypoints_data/incorrect_mirrored.csv"
INCORRECT_JITTER_CSV_PATH = "keypoints_data/incorrect_jittered.csv"
INCORRECT_MIRROR_JITTER_CSV_PATH = "keypoints_data/incorrect_mirrored_jittered.csv"

# Model save path
MODEL_SAVE_PATH = "models/squat_pose_model.h5"

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
DROPOUT_RATE = 0.3

# =============================================================================
# SETUP AND LOGGING
# =============================================================================

def setup_logging():
    """Configure logging for training progress."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    logging.info(f"Ensured directory exists: {os.path.dirname(MODEL_SAVE_PATH)}")

# =============================================================================
# DATA HANDLING
# =============================================================================

def load_and_prepare_data():
    """
    Load all CSV files, read label from the first column (0/1),
    use the remaining 51 columns as features, merge, shuffle, and split.
    
    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    logging.info("Loading and preparing training data...")

    # Load 4 files per class: original, mirrored, original_jittered, mirrored_jittered
    data_paths = [
        CORRECT_ORIG_CSV_PATH,
        CORRECT_MIRROR_CSV_PATH,
        CORRECT_JITTER_CSV_PATH,
        CORRECT_MIRROR_JITTER_CSV_PATH,
        INCORRECT_ORIG_CSV_PATH,
        INCORRECT_MIRROR_CSV_PATH,
        INCORRECT_JITTER_CSV_PATH,
        INCORRECT_MIRROR_JITTER_CSV_PATH,
    ]

    feature_blocks = []  # list of np.ndarrays (N_i, 51)
    label_blocks = []    # list of np.ndarrays (N_i,)

    total_loaded = 0
    for path in data_paths:
        if not os.path.exists(path):
            logging.warning(f"File not found: {path}")
            continue
        try:
            df = pd.read_csv(path)
            if df.shape[1] < 52:
                raise ValueError(f"Expected at least 52 columns (1 label + 51 features) in {path}, found {df.shape[1]}")

            # First column is label, remaining 51 columns are features
            y = df.iloc[:, 0].to_numpy().astype(np.float32)
            X = df.iloc[:, 1:52].to_numpy().astype(np.float32)

            feature_blocks.append(X)
            label_blocks.append(y)
            total_loaded += len(df)
            logging.info(f"Loaded {len(df)} samples from {path}")
        except Exception as e:
            logging.error(f"Error loading {path}: {e}")

    if not feature_blocks:
        raise ValueError("No data files could be loaded or contained no samples.")

    # Merge all
    X_all = np.vstack(feature_blocks)
    y_all = np.concatenate(label_blocks)

    # Log distribution
    try:
        counts = np.bincount(y_all.astype(np.int32))
        logging.info(f"Combined dataset: {len(y_all)} total samples")
        logging.info(f"Label distribution (0,1): {counts}")
    except Exception:
        logging.info(f"Combined dataset: {len(y_all)} total samples")

    # Shuffle while keeping X/y aligned
    indices = np.arange(len(y_all))
    np.random.shuffle(indices)
    X_all = X_all[indices]
    y_all = y_all[indices]

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=y_all
    )

    logging.info(f"Training set: {len(X_train)} samples")
    logging.info(f"Validation set: {len(X_val)} samples")

    return X_train, X_val, y_train, y_val

def prepare_tensorflow_data(X_train, X_val, y_train, y_val):
    """
    Convert pandas DataFrames to TensorFlow datasets.
    
    Args:
        X_train, X_val: pandas DataFrames with keypoint data
        y_train, y_val: numpy arrays with labels
    
    Returns:
        tuple: (train_dataset, val_dataset) - TensorFlow datasets
    """
    logging.info("Preparing TensorFlow datasets...")
    
    # Convert to numpy arrays (assuming keypoints are in columns)
    # The exact column structure depends on your CSV format
    # This assumes keypoints are stored as 51 columns (17 joints * 3 values each)
    
    def dataframe_to_keypoints(df):
        """Convert DataFrame to keypoint tensor."""
        # Assuming the first 51 columns are keypoints [y1, x1, conf1, y2, x2, conf2, ...]
        keypoint_cols = [col for col in df.columns if col.startswith(('y', 'x', 'conf'))]
        if len(keypoint_cols) == 51:
            keypoints = df[keypoint_cols].values.astype(np.float32)
        else:
            # Fallback: assume all numeric columns are keypoints
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 51:
                keypoints = df[numeric_cols[:51]].values.astype(np.float32)
            else:
                raise ValueError(f"Expected 51 keypoint columns, found {len(numeric_cols)}")
        return keypoints
    
    X_train_keypoints = dataframe_to_keypoints(X_train)
    X_val_keypoints = dataframe_to_keypoints(X_val)
    
    # Convert labels to float32 for binary crossentropy
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    
    # Create dictionary labels for standardized 3-output CNN architecture
    # All CNNs output: form_score, instruction_id, joint_masked_keypoints
    # Only form_score is trained; others are for inference only
    def create_standardized_labels(labels):
        """Create standardized dictionary labels for 3-output CNN training."""
        batch_size = len(labels)
        return {
            'form_score': labels,  # Actual training labels (0.0 or 1.0)
            'instruction_id': np.zeros(batch_size, dtype=np.int32),  # Placeholder for inference
            'joint_masked_keypoints': np.zeros((batch_size, 17, 3), dtype=np.float32)  # Placeholder for inference
        }
    
    # Create standardized dictionary labels
    y_train_dict = create_standardized_labels(y_train)
    y_val_dict = create_standardized_labels(y_val)
    
    logging.info("Creating standardized 3-output dataset structure:")
    logging.info(f"  form_score: {y_train_dict['form_score'].shape} (trainable)")
    logging.info(f"  instruction_id: {y_train_dict['instruction_id'].shape} (inference only)")
    logging.info(f"  joint_masked_keypoints: {y_train_dict['joint_masked_keypoints'].shape} (inference only)")
    
    # Create TensorFlow datasets with dictionary outputs
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_keypoints, y_train_dict))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_keypoints, y_val_dict))
    
    # Batch and prefetch for performance
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    logging.info(f"Keypoint shape: {X_train_keypoints.shape[1:]}")
    logging.info("TensorFlow datasets prepared successfully for standardized 3-output CNN architecture")
    
    return train_dataset, val_dataset

# =============================================================================
# MODEL IMPORT AND BUILDING
# =============================================================================

def import_cnn_module(module_name):
    """
    Dynamically import the specified CNN module.
    
    Args:
        module_name (str): Module path (e.g., "cnns.squat_pose_cnn")
    
    Returns:
        module: Imported module
    """
    try:
        logging.info(f"Importing CNN module: {module_name}")
        module = importlib.import_module(module_name)
        logging.info(f"Successfully imported {module_name}")
        return module
    except ImportError as e:
        logging.error(f"Failed to import {module_name}: {e}")
        raise

def build_model_from_module(module, dropout_rate=0.3):
    """
    Build and compile a model using the module's build_model function.
    
    Args:
        module: Imported CNN module
        dropout_rate (float): Dropout rate for regularization
    
    Returns:
        tf.keras.Model: Compiled model
    """
    try:
        # Use the module's build_model function (all CNNs will have this)
        logging.info("Using module.build_model() function")
        model = module.build_model(dropout_rate=dropout_rate)
        
        logging.info("Model built and compiled successfully for standardized 3-output CNN architecture")
        return model
        
    except Exception as e:
        logging.error(f"Error building model: {e}")
        raise

# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, train_dataset, val_dataset):
    """
    Train the model with the specified configuration.
    
    Args:
        model: Compiled Keras model
        train_dataset: Training dataset
        val_dataset: Validation dataset
    
    Returns:
        tf.keras.callbacks.History: Training history
    """
    logging.info("Starting model training...")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH.replace('.h5', '_checkpoint.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Training
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    logging.info("Training completed successfully")
    return history

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main training pipeline."""
    logger = setup_logging()
    
    try:
        # Create necessary directories
        create_directories()
        
        # Load and prepare data
        X_train, X_val, y_train, y_val = load_and_prepare_data()
        train_dataset, val_dataset = prepare_tensorflow_data(X_train, X_val, y_train, y_val)
        
        # Import CNN module
        cnn_module = import_cnn_module(CNN_MODULE)
        
        # Build model
        model = build_model_from_module(cnn_module, dropout_rate=DROPOUT_RATE)
        
        # Display model summary
        logging.info("Model architecture:")
        model.summary()
        
        # Display training configuration
        logging.info("=" * 60)
        logging.info("TRAINING CONFIGURATION")
        logging.info("=" * 60)
        logging.info("✓ Standardized 3-output CNN architecture")
        logging.info("✓ Training focused on 'form_score' output only")
        logging.info("✓ 'instruction_id' and 'joint_masked_keypoints' preserved for inference")
        logging.info("✓ Loss: binary_crossentropy for form_score")
        logging.info("✓ Metrics: accuracy for form_score")
        logging.info("✓ Other outputs: minimal training impact (loss_weight = 0.0)")
        logging.info("=" * 60)
        
        # Train model
        history = train_model(model, train_dataset, val_dataset)
        
        # Save final model
        model.save(MODEL_SAVE_PATH)
        logging.info(f"Model saved successfully to: {MODEL_SAVE_PATH}")
        
        # Print final metrics
        val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
        logging.info(f"Final validation accuracy: {val_accuracy:.4f}")
        logging.info(f"Final validation loss: {val_loss:.4f}")
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Model saved to: {MODEL_SAVE_PATH}")
        print(f"Final validation accuracy: {val_accuracy:.4f}")
        print(f"Final validation loss: {val_loss:.4f}")
        print(f"{'='*60}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        print(f"ERROR: Training failed - {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()