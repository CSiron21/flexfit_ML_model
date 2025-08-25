"""FlexFit ML Model Training Script - Trains pose analysis CNN models with configurable architecture."""

import os
import sys
import importlib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging for CPU training

# Configuration
CNN_MODULE = "cnns.overhead_pose_cnn"
CORRECT_DIR = "keypoints_data/overhead_presses/correct"
INCORRECT_DIR = "keypoints_data/overhead_presses/incorrect"
MODEL_SAVE_PATH = "models/overhead_pose_model.keras"
TFLITE_SAVE_PATH = "models/overhead_pose_float16.tflite"
TRAINING_LOG_PATH = "training_logs/overhead_pose_training.log"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0005
VALIDATION_SPLIT = 0.2
DROPOUT_RATE = 0.4

def setup_logging():
	"""Configure logging for training progress."""
	os.makedirs(os.path.dirname(TRAINING_LOG_PATH), exist_ok=True)
	
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s',
		handlers=[
			logging.FileHandler(TRAINING_LOG_PATH),
			logging.StreamHandler(sys.stdout)
		]
	)
	return logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    logging.info(f"Ensured directory exists: {os.path.dirname(MODEL_SAVE_PATH)}")

def _list_csv_files(directory: str) -> list:
    if not directory or not os.path.isdir(directory):
        logging.warning(f"Directory not found or invalid: {directory}")
        return []
    files = []
    for name in os.listdir(directory):
        if name.lower().endswith('.csv'):
            files.append(os.path.join(directory, name))
    return sorted(files)

def load_and_prepare_data():
    """Load CSVs from correct/incorrect directories and prepare train/val split."""
    logging.info("Loading and preparing training data...")

    correct_files = _list_csv_files(CORRECT_DIR)
    incorrect_files = _list_csv_files(INCORRECT_DIR)

    if not correct_files and not incorrect_files:
        raise ValueError("No CSV files found in the provided directories.")

    feature_blocks = []
    label_blocks = []

    def _ingest(files: list):
        loaded = 0
        for path in files:
            try:
                df = pd.read_csv(path, header=0)  # First row is headers
                if df.empty:
                    logging.warning(f"Empty CSV skipped: {path}")
                    continue
                if df.shape[1] != 52:
                    raise ValueError(f"{path}: expected 52 columns (label + 51 keypoints), found {df.shape[1]}")
                
                y = df.iloc[:, 0].to_numpy().astype(np.float32)  # First column: label
                X = df.iloc[:, 1:52].to_numpy().astype(np.float32)  # Next 51 columns: keypoints
                feature_blocks.append(X)
                label_blocks.append(y)
                loaded += len(df)
                logging.info(f"Loaded {len(df)} samples from {path}")
            except Exception as e:
                logging.error(f"Error loading {path}: {e}")
        return loaded

    n_correct = _ingest(correct_files)
    n_incorrect = _ingest(incorrect_files)

    if not feature_blocks:
        raise ValueError("No valid samples ingested from the directories.")

    # Merge all
    X_all = np.vstack(feature_blocks)
    y_all = np.concatenate(label_blocks)

    try:
        counts = np.bincount(y_all.astype(np.int32))
        logging.info(f"Combined dataset: {len(y_all)} total samples (correct={n_correct}, incorrect={n_incorrect})")
        logging.info(f"Label distribution (0,1): {counts}")
    except Exception:
        logging.info(f"Combined dataset: {len(y_all)} total samples (correct={n_correct}, incorrect={n_incorrect})")

    indices = np.arange(len(y_all))
    np.random.shuffle(indices)
    X_all = X_all[indices]
    y_all = y_all[indices]

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=y_all
    )

    logging.info(f"Training set: {len(X_train)} samples")
    logging.info(f"Validation set: {len(X_val)} samples")
    
    try:
        train_counts = np.bincount(y_train.astype(np.int32))
        val_counts = np.bincount(y_val.astype(np.int32))
        logging.info(f"Training set class distribution (0,1): {train_counts}")
        logging.info(f"Validation set class distribution (0,1): {val_counts}")
        logging.info(f"Training set class balance: {train_counts[1]/(train_counts[0]+train_counts[1]):.3f} (1s)")
        logging.info(f"Validation set class balance: {val_counts[1]/(val_counts[0]+val_counts[1]):.3f} (1s)")
    except Exception as e:
        logging.warning(f"Could not analyze class distribution: {e}")

    return X_train, X_val, y_train, y_val

def prepare_tensorflow_data(X_train, X_val, y_train, y_val):
    """Convert numpy arrays to TensorFlow datasets."""
    logging.info("Preparing TensorFlow datasets...")
    
    X_train_keypoints = X_train.astype(np.float32)
    X_val_keypoints = X_val.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    
    def create_standardized_labels(labels):
        """Create standardized dictionary labels for 3-output CNN training."""
        batch_size = len(labels)
        return {
            'form_score': labels.reshape(-1, 1),
            'instruction_id': np.zeros(batch_size, dtype=np.int32),
            'joint_masked_keypoints': np.zeros((batch_size, 17, 3), dtype=np.float32)
        }
    
    y_train_dict = create_standardized_labels(y_train)
    y_val_dict = create_standardized_labels(y_val)
    
    logging.info("Creating standardized 3-output dataset structure:")
    logging.info(f"  form_score: {y_train_dict['form_score'].shape} (trainable)")
    logging.info(f"  instruction_id: {y_train_dict['instruction_id'].shape} (inference only)")
    logging.info(f"  joint_masked_keypoints: {y_train_dict['joint_masked_keypoints'].shape} (inference only)")
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_keypoints, y_train_dict))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_keypoints, y_val_dict))
    
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(1)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(1)
    
    logging.info(f"Keypoint shape: {X_train_keypoints.shape[1:]}")
    logging.info("TensorFlow datasets prepared successfully for standardized 3-output CNN architecture")
    
    return train_dataset, val_dataset

def import_and_build_model(module_name, dropout_rate=0.3, learning_rate=0.001):
    """Import CNN module and build model using module's build_model function."""
    try:
        logging.info(f"Importing CNN module: {module_name}")
        cnn_module = importlib.import_module(module_name)
        logging.info(f"Successfully imported {module_name}")
        
        logging.info("Building model using module.build_model()")
        model = cnn_module.build_model(dropout_rate=dropout_rate, learning_rate=learning_rate)
        
        logging.info("Model built and compiled successfully for standardized 3-output CNN architecture")
        return model
        
    except Exception as e:
        logging.error(f"Error importing/building model: {e}")
        raise

def train_model(model, train_dataset, val_dataset):
    """Train the model with the specified configuration."""
    logging.info("Starting model training...")
    
    callbacks = [
        EarlyStopping(
            monitor='val_form_score_loss',
            mode='min',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH.replace('.keras', '_checkpoint.keras'),
            monitor='val_form_score_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    logging.info("Training completed successfully")
    return history

def main():
    """Main training pipeline."""
    logger = setup_logging()
    
    try:
        # Create necessary directories
        create_directories()
        
        # Load and prepare data
        X_train, X_val, y_train, y_val = load_and_prepare_data()
        train_dataset, val_dataset = prepare_tensorflow_data(X_train, X_val, y_train, y_val)
        
        # Import CNN module and build model
        model = import_and_build_model(CNN_MODULE, dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE)
        
        model.summary()
        
        history = train_model(model, train_dataset, val_dataset)
        
        model.save(MODEL_SAVE_PATH)
        logging.info(f"Model saved successfully to: {MODEL_SAVE_PATH}")
        
        results = model.evaluate(val_dataset, verbose=0)
        
        logging.info("Model evaluation results order:")
        logging.info(f"  results[0] = total_loss: {results[0]:.4f}")
        logging.info(f"  results[1] = form_score_loss: {results[1]:.4f}")
        logging.info(f"  results[2] = instruction_id_loss: {results[2]:.4f}")
        logging.info(f"  results[3] = joint_masked_keypoints_loss: {results[3]:.4f}")
        logging.info(f"  results[4] = form_score_accuracy: {results[4]:.4f}")
        
        form_score_loss = results[1]
        form_score_accuracy = results[4]
        
        logging.info(f"Final form_score validation accuracy (best checkpoint): {form_score_accuracy:.4f}")
        logging.info(f"Final form_score validation loss (best checkpoint): {form_score_loss:.4f}")
        

        logging.info("Starting TFLite conversion from best checkpoint model...")
        
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            logging.info("Converting best checkpoint model to TFLite format...")
            tflite_model = converter.convert()
            
            with open(TFLITE_SAVE_PATH, "wb") as f:
                f.write(tflite_model)
            
            tflite_size = len(tflite_model) / (1024 * 1024)
            logging.info(f"TFLite conversion successful! Model size: {tflite_size:.2f} MB")
            
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Model saved to: {MODEL_SAVE_PATH}")
            print(f"Best checkpoint form_score accuracy: {form_score_accuracy:.4f}")
            print(f"Best checkpoint form_score loss: {form_score_loss:.4f}")
            print(f"{'='*60}")
            print(f"TFLITE CONVERSION SUCCESSFUL!")
            print(f"TFLite model saved to: {TFLITE_SAVE_PATH}")
            print(f"TFLite model size: {tflite_size:.2f} MB")
            print(f"{'='*60}")
            
        except Exception as e:
            logging.error(f"TFLite conversion failed: {e}")
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Model saved to: {MODEL_SAVE_PATH}")
            print(f"Best checkpoint form_score accuracy: {form_score_accuracy:.4f}")
            print(f"Best checkpoint form_score loss: {form_score_loss:.4f}")
            print(f"{'='*60}")
            print(f"WARNING: TFLite conversion failed - {e}")
            print(f"Keras model is still available for manual conversion")
            print(f"{'='*60}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        print(f"ERROR: Training failed - {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()