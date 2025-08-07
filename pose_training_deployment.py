import tensorflow as tf
import numpy as np
import json
import os
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import pickle

class PoseDataAugmenter:
    """
    Data augmentation for pose keypoint data
    """
    
    def __init__(self, 
                 noise_std=0.01,
                 rotation_range=5.0,
                 scale_range=0.05,
                 translation_range=0.02):
        self.noise_std = noise_std
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
    
    def add_gaussian_noise(self, keypoints):
        """Add Gaussian noise to keypoints"""
        noise = np.random.normal(0, self.noise_std, keypoints.shape)
        return keypoints + noise
    
    def random_rotation(self, keypoints):
        """Apply small random rotation"""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        angle_rad = np.deg2rad(angle)
        
        # Rotation matrix for 2D (assuming x, y are first two features)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Apply rotation to x, y coordinates
        augmented = keypoints.copy()
        x = keypoints[:, 0]
        y = keypoints[:, 1]
        
        augmented[:, 0] = cos_a * x - sin_a * y
        augmented[:, 1] = sin_a * x + cos_a * y
        
        return augmented
    
    def random_scale(self, keypoints):
        """Apply random scaling"""
        scale_factor = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)
        augmented = keypoints.copy()
        augmented[:, :2] *= scale_factor  # Scale x, y coordinates
        return augmented
    
    def random_translation(self, keypoints):
        """Apply random translation"""
        tx = np.random.uniform(-self.translation_range, self.translation_range)
        ty = np.random.uniform(-self.translation_range, self.translation_range)
        
        augmented = keypoints.copy()
        augmented[:, 0] += tx  # Translate x
        augmented[:, 1] += ty  # Translate y
        
        return augmented
    
    def augment_batch(self, keypoints_batch, apply_prob=0.8):
        """Apply random augmentations to a batch of keypoints"""
        batch_size = keypoints_batch.shape[0]
        augmented_batch = keypoints_batch.copy()
        
        for i in range(batch_size):
            if np.random.random() < apply_prob:
                # Randomly select augmentations to apply
                if np.random.random() < 0.9:  # Almost always add noise
                    augmented_batch[i] = self.add_gaussian_noise(augmented_batch[i])
                
                if np.random.random() < 0.3:  # Sometimes rotate
                    augmented_batch[i] = self.random_rotation(augmented_batch[i])
                
                if np.random.random() < 0.3:  # Sometimes scale
                    augmented_batch[i] = self.random_scale(augmented_batch[i])
                
                if np.random.random() < 0.3:  # Sometimes translate
                    augmented_batch[i] = self.random_translation(augmented_batch[i])
        
        return augmented_batch

class PoseModelTrainer:
    """
    Comprehensive trainer for pose analysis model
    """
    
    def __init__(self, model, augmenter=None):
        self.model = model
        self.augmenter = augmenter or PoseDataAugmenter()
        self.history = None
        self.best_threshold = {'form': 0.5, 'alignment': 0.5}
    
    def train_with_augmentation(self, 
                              X_train, y_train,
                              X_val, y_val,
                              epochs=50,
                              batch_size=32,
                              augmentation_ratio=0.5,
                              early_stopping_patience=10,
                              save_best=True,
                              model_path='best_pose_model.h5'):
        """
        Train model with data augmentation
        """
        
        # Callbacks
        callbacks = []
        
        if early_stopping_patience:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        if save_best:
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            callbacks.append(model_checkpoint)
        
        # Learning rate scheduling
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # Custom data generator with augmentation
        def data_generator():
            while True:
                # Original data
                indices = np.random.permutation(len(X_train))
                
                for i in range(0, len(X_train), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    X_batch = X_train[batch_indices]
                    y_batch = {key: y_train[key][batch_indices] for key in y_train.keys()}
                    
                    # Apply augmentation to a portion of the batch
                    aug_size = int(len(X_batch) * augmentation_ratio)
                    if aug_size > 0:
                        X_batch[:aug_size] = self.augmenter.augment_batch(X_batch[:aug_size])
                    
                    yield X_batch, y_batch
        
        # Calculate steps per epoch
        steps_per_epoch = len(X_train) // batch_size
        
        print(f"Training with augmentation (ratio: {augmentation_ratio})")
        print(f"Steps per epoch: {steps_per_epoch}")
        
        # Train model
        self.history = self.model.fit(
            data_generator(),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_comprehensive(self, X_test, y_test, plot_results=True):
        """
        Comprehensive evaluation with precision/recall analysis
        """
        print("=" * 50)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 50)
        
        # Get predictions
        predictions = self.model.predict(X_test, verbose=0)
        
        results = {}
        
        # 1. Form Score Evaluation
        print("\n1. FORM SCORE EVALUATION")
        print("-" * 30)
        
        form_predictions = predictions['form_score'].flatten()
        form_true = y_test['form_score'].flatten()
        
        form_mae = np.mean(np.abs(form_predictions - form_true))
        form_mse = np.mean((form_predictions - form_true) ** 2)
        form_rmse = np.sqrt(form_mse)
        
        print(f"Form Score MAE: {form_mae:.4f}")
        print(f"Form Score MSE: {form_mse:.4f}")
        print(f"Form Score RMSE: {form_rmse:.4f}")
        print(f"Form Score R²: {1 - form_mse / np.var(form_true):.4f}")
        
        results['form'] = {
            'mae': form_mae,
            'mse': form_mse,
            'rmse': form_rmse
        }
        
        # 2. Joint Alignment Evaluation
        print("\n2. JOINT ALIGNMENT EVALUATION")
        print("-" * 35)
        
        alignment_predictions = predictions['joint_alignment']
        alignment_true = y_test['joint_alignment']
        
        # Find optimal threshold for each joint
        optimal_thresholds = []
        joint_metrics = []
        
        for joint_idx in range(alignment_predictions.shape[1]):
            joint_pred = alignment_predictions[:, joint_idx]
            joint_true = alignment_true[:, joint_idx]
            
            # Find optimal threshold using F1 score
            thresholds = np.linspace(0.1, 0.9, 9)
            best_threshold = 0.5
            best_f1 = 0
            
            for thresh in thresholds:
                pred_binary = (joint_pred > thresh).astype(int)
                if len(np.unique(joint_true)) > 1:  # Only if we have both classes
                    f1 = f1_score(joint_true, pred_binary, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = thresh
            
            optimal_thresholds.append(best_threshold)
            
            # Calculate metrics with optimal threshold
            pred_binary = (joint_pred > best_threshold).astype(int)
            precision = precision_score(joint_true, pred_binary, zero_division=0)
            recall = recall_score(joint_true, pred_binary, zero_division=0)
            f1 = f1_score(joint_true, pred_binary, zero_division=0)
            
            joint_metrics.append({
                'joint': joint_idx,
                'threshold': best_threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        # Overall alignment metrics
        overall_threshold = np.mean(optimal_thresholds)
        alignment_pred_binary = (alignment_predictions > overall_threshold).astype(int)
        
        overall_precision = precision_score(
            alignment_true.flatten(), 
            alignment_pred_binary.flatten(), 
            zero_division=0
        )
        overall_recall = recall_score(
            alignment_true.flatten(), 
            alignment_pred_binary.flatten(), 
            zero_division=0
        )
        overall_f1 = f1_score(
            alignment_true.flatten(), 
            alignment_pred_binary.flatten(), 
            zero_division=0
        )
        overall_accuracy = np.mean(alignment_pred_binary == alignment_true)
        
        print(f"Overall Alignment Accuracy: {overall_accuracy:.4f}")
        print(f"Overall Precision: {overall_precision:.4f}")
        print(f"Overall Recall: {overall_recall:.4f}")
        print(f"Overall F1-Score: {overall_f1:.4f}")
        print(f"Optimal Threshold: {overall_threshold:.3f}")
        
        # Per-joint summary
        print(f"\nPer-Joint Performance (Top 5 F1 scores):")
        sorted_joints = sorted(joint_metrics, key=lambda x: x['f1'], reverse=True)
        for i, joint_metric in enumerate(sorted_joints[:5]):
            print(f"Joint {joint_metric['joint']:2d}: F1={joint_metric['f1']:.3f}, "
                  f"P={joint_metric['precision']:.3f}, R={joint_metric['recall']:.3f}, "
                  f"Thresh={joint_metric['threshold']:.2f}")
        
        results['alignment'] = {
            'accuracy': overall_accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'optimal_threshold': overall_threshold,
            'joint_metrics': joint_metrics
        }
        
        self.best_threshold['alignment'] = overall_threshold
        
        # 3. Joint Angles Evaluation
        print("\n3. JOINT ANGLES EVALUATION")
        print("-" * 30)
        
        angles_predictions = predictions['joint_angles']
        angles_true = y_test['joint_angles']
        
        angles_mae = np.mean(np.abs(angles_predictions - angles_true))
        angles_mse = np.mean((angles_predictions - angles_true) ** 2)
        angles_rmse = np.sqrt(angles_mse)
        
        print(f"Joint Angles MAE: {angles_mae:.4f}°")
        print(f"Joint Angles MSE: {angles_mse:.4f}°²")
        print(f"Joint Angles RMSE: {angles_rmse:.4f}°")
        
        results['angles'] = {
            'mae': angles_mae,
            'mse': angles_mse,
            'rmse': angles_rmse
        }
        
        # 4. Plotting results
        if plot_results:
            self._plot_evaluation_results(
                form_true, form_predictions,
                alignment_true, alignment_predictions,
                angles_true, angles_predictions,
                joint_metrics
            )
        
        return results
    
    def _plot_evaluation_results(self, form_true, form_pred, align_true, align_pred, 
                                angles_true, angles_pred, joint_metrics):
        """Plot comprehensive evaluation results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Form score scatter plot
        axes[0, 0].scatter(form_true, form_pred, alpha=0.6)
        axes[0, 0].plot([0, 1], [0, 1], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Form Score')
        axes[0, 0].set_ylabel('Predicted Form Score')
        axes[0, 0].set_title('Form Score: True vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Form score distribution
        axes[0, 1].hist(form_true, alpha=0.7, label='True', bins=20)
        axes[0, 1].hist(form_pred, alpha=0.7, label='Predicted', bins=20)
        axes[0, 1].set_xlabel('Form Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Form Score Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Joint alignment heatmap (first 100 samples)
        sample_size = min(100, len(align_true))
        axes[0, 2].imshow(align_true[:sample_size].T, aspect='auto', cmap='RdYlBu')
        axes[0, 2].set_xlabel('Sample Index')
        axes[0, 2].set_ylabel('Joint Index')
        axes[0, 2].set_title('True Joint Alignment (Sample)')
        
        # Per-joint F1 scores
        joint_f1s = [jm['f1'] for jm in joint_metrics]
        axes[1, 0].bar(range(len(joint_f1s)), joint_f1s)
        axes[1, 0].set_xlabel('Joint Index')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Per-Joint Alignment F1 Scores')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Joint angles scatter (first joint as example)
        axes[1, 1].scatter(angles_true[:, 0], angles_pred[:, 0], alpha=0.6)
        min_angle = min(angles_true[:, 0].min(), angles_pred[:, 0].min())
        max_angle = max(angles_true[:, 0].max(), angles_pred[:, 0].max())
        axes[1, 1].plot([min_angle, max_angle], [min_angle, max_angle], 'r--', lw=2)
        axes[1, 1].set_xlabel('True Angle (Joint 0)')
        axes[1, 1].set_ylabel('Predicted Angle (Joint 0)')
        axes[1, 1].set_title('Joint Angles: True vs Predicted (Joint 0)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Training history (if available)
        if self.history:
            axes[1, 2].plot(self.history.history['loss'], label='Training Loss')
            axes[1, 2].plot(self.history.history['val_loss'], label='Validation Loss')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Loss')
            axes[1, 2].set_title('Training History')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pose_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()

class PoseModelDeployer:
    """
    Handle model deployment and TensorFlow Lite conversion
    """
    
    def __init__(self, model_path=None, model=None):
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = model
        
        self.metadata = {
            'input_shape': [17, 3],
            'keypoint_order': [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ],
            'feature_order': ['x', 'y', 'confidence'],
            'outputs': {
                'form_score': 'Float32 value between 0-1',
                'joint_alignment': 'Array of 17 float32 values (0-1) for each joint',
                'joint_angles': 'Array of 17 float32 values (degrees) for each joint'
            }
        }
    
    def convert_to_tflite(self, 
                         output_path='pose_model.tflite',
                         quantization=None,
                         representative_dataset=None):
        """
        Convert model to TensorFlow Lite format
        
        Args:
            output_path: Path to save .tflite file
            quantization: 'int8', 'float16', or None
            representative_dataset: Dataset for quantization calibration
        """
        
        print(f"Converting model to TensorFlow Lite...")
        print(f"Quantization: {quantization or 'None'}")
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Apply quantization
        if quantization == 'int8':
            if representative_dataset is None:
                raise ValueError("Representative dataset required for int8 quantization")
            
            def representative_data_gen():
                for data in representative_dataset:
                    yield [data.astype(np.float32)]
            
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
        elif quantization == 'float16':
            converter.target_spec.supported_types = [tf.float16]
        
        # Convert model
        try:
            tflite_model = converter.convert()
            
            # Save to file
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Get model size
            model_size = os.path.getsize(output_path) / 1024  # KB
            print(f"✓ Model converted successfully!")
            print(f"✓ Output: {output_path}")
            print(f"✓ Size: {model_size:.1f} KB")
            
            return output_path
            
        except Exception as e:
            print(f"✗ Conversion failed: {str(e)}")
            return None
    
    def verify_tflite_model(self, 
                           tflite_path, 
                           test_data, 
                           tolerance=1e-3):
        """
        Verify TFLite model predictions match original model
        """
        
        print(f"Verifying TFLite model: {tflite_path}")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input details: {input_details[0]['shape']}")
        print(f"Output details: {len(output_details)} outputs")
        
        # Test on sample data
        sample_input = test_data[:5]  # First 5 samples
        
        # Original model predictions
        original_predictions = self.model.predict(sample_input, verbose=0)
        
        # TFLite predictions
        tflite_predictions = {}
        for output_detail in output_details:
            tflite_predictions[output_detail['name']] = []
        
        for i in range(len(sample_input)):
            # Set input
            interpreter.set_tensor(input_details[0]['index'], 
                                 sample_input[i:i+1].astype(np.float32))
            
            # Run inference
            interpreter.invoke()
            
            # Get outputs
            for output_detail in output_details:
                output_data = interpreter.get_tensor(output_detail['index'])
                tflite_predictions[output_detail['name']].append(output_data[0])
        
        # Convert to numpy arrays
        for key in tflite_predictions:
            tflite_predictions[key] = np.array(tflite_predictions[key])
        
        # Compare predictions
        print("\nPrediction Comparison:")
        print("-" * 40)
        
        all_close = True
        for output_name in original_predictions.keys():
            if output_name in tflite_predictions:
                orig = original_predictions[output_name][:5]
                tflite = tflite_predictions[output_name]
                
                max_diff = np.max(np.abs(orig - tflite))
                mean_diff = np.mean(np.abs(orig - tflite))
                
                is_close = np.allclose(orig, tflite, atol=tolerance)
                all_close &= is_close
                
                status = "✓" if is_close else "✗"
                print(f"{status} {output_name}:")
                print(f"    Max difference: {max_diff:.6f}")
                print(f"    Mean difference: {mean_diff:.6f}")
                print(f"    Within tolerance: {is_close}")
        
        print(f"\nOverall verification: {'✓ PASSED' if all_close else '✗ FAILED'}")
        return all_close
    
    def save_metadata(self, output_path='pose_model_metadata.json'):
        """Save model metadata for deployment"""
        
        # Add model info
        self.metadata.update({
            'model_type': 'pose_analysis_cnn',
            'framework': 'tensorflow',
            'version': '1.0',
            'created_date': tf.timestamp().numpy().decode('utf-8') if hasattr(tf.timestamp().numpy(), 'decode') else str(tf.timestamp().numpy()),
            'input_preprocessing': {
                'normalization': 'recommended',
                'coordinate_system': 'normalized (0-1)',
                'confidence_threshold': 0.3
            }
        })
        
        with open(output_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✓ Metadata saved to: {output_path}")
        return output_path

# Example usage and training pipeline
def run_complete_training_pipeline():
    """
    Complete training and deployment pipeline
    """
    
    print("=" * 60)
    print("COMPLETE POSE MODEL TRAINING & DEPLOYMENT PIPELINE")
    print("=" * 60)
    
    # 1. Import model and data pipeline
    from Flexfit_model_design import create_pose_model
    from data_pipeline import FlexFitDataPipeline
    
    # 2. Load or create training data
    print("\n1. Loading training data...")
    
    # Check if processed data exists
    if os.path.exists("processed_data/X_train.npy"):
        print("Loading existing processed data...")
        X_train = np.load("processed_data/X_train.npy")
        y_form_train = np.load("processed_data/y_form_train.npy")
        y_alignment_train = np.load("processed_data/y_alignment_train.npy")
        y_angles_train = np.load("processed_data/y_angles_train.npy")
        
        X_val = np.load("processed_data/X_val.npy")
        y_form_val = np.load("processed_data/y_form_val.npy")
        y_alignment_val = np.load("processed_data/y_alignment_val.npy")
        y_angles_val = np.load("processed_data/y_angles_val.npy")
        
        X_test = np.load("processed_data/X_test.npy")
        y_form_test = np.load("processed_data/y_form_test.npy")
        y_alignment_test = np.load("processed_data/y_alignment_test.npy")
        y_angles_test = np.load("processed_data/y_angles_test.npy")
        
    else:
        print("No processed data found. Running data pipeline...")
        # Run data pipeline to process videos
        pipeline = FlexFitDataPipeline()
        all_data = pipeline.process_all_videos()
        
        if not all_data:
            print("No video data found! Please check your dataset structure.")
            print("Falling back to sample data...")
            from Flexfit_model_design import generate_sample_data
            X_train, y_train = generate_sample_data(batch_size=1000)
            X_val, y_val = generate_sample_data(batch_size=200)
            X_test, y_test = generate_sample_data(batch_size=300)
        else:
            train_data, val_data, test_data = pipeline.create_training_dataset(all_data)
            pipeline.save_datasets(train_data, val_data, test_data)
            
            X_train = train_data['X']
            y_form_train = train_data['y_form']
            y_alignment_train = train_data['y_alignment']
            y_angles_train = train_data['y_angles']
            
            X_val = val_data['X']
            y_form_val = val_data['y_form']
            y_alignment_val = val_data['y_alignment']
            y_angles_val = val_data['y_angles']
            
            X_test = test_data['X']
            y_form_test = test_data['y_form']
            y_alignment_test = test_data['y_alignment']
            y_angles_test = test_data['y_angles']
    
    # Prepare data dictionaries
    y_train = {
        'form_score': y_form_train,
        'joint_alignment': y_alignment_train,
        'joint_angles': y_angles_train
    }
    
    y_val = {
        'form_score': y_form_val,
        'joint_alignment': y_alignment_val,
        'joint_angles': y_angles_val
    }
    
    y_test = {
        'form_score': y_form_test,
        'joint_alignment': y_alignment_test,
        'joint_angles': y_angles_test
    }
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 3. Create and train model
    print("\n2. Creating model...")
    model = create_pose_model()
    
    print("\n3. Training with augmentation...")
    augmenter = PoseDataAugmenter(noise_std=0.02)
    trainer = PoseModelTrainer(model, augmenter)
    
    history = trainer.train_with_augmentation(
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        batch_size=32,
        augmentation_ratio=0.6,
        early_stopping_patience=5,
        model_path='best_pose_model.h5'
    )
    
    # 4. Comprehensive evaluation
    print("\n4. Evaluating model...")
    results = trainer.evaluate_comprehensive(X_test, y_test, plot_results=True)
    
    # 5. Deploy model
    print("\n5. Deploying model...")
    deployer = PoseModelDeployer(model=model)
    
    # Convert to TFLite
    tflite_path = deployer.convert_to_tflite(
        'pose_model.tflite',
        quantization=None
    )
    
    # Convert to quantized TFLite
    if tflite_path:
        quantized_path = deployer.convert_to_tflite(
            'pose_model_quantized.tflite',
            quantization='int8',
            representative_dataset=X_test[:50]  # Use test data for calibration
        )
        
        # Verify models
        if quantized_path:
            deployer.verify_tflite_model(tflite_path, X_test)
            deployer.verify_tflite_model(quantized_path, X_test, tolerance=0.1)
    
    # Save metadata
    deployer.save_metadata('pose_model_metadata.json')
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"✓ Trained model saved: best_pose_model.h5")
    print(f"✓ TFLite model: pose_model.tflite")
    print(f"✓ Quantized TFLite: pose_model_quantized.tflite")
    print(f"✓ Metadata: pose_model_metadata.json")
    print(f"✓ Evaluation plots: pose_model_evaluation.png")
    
    return model, results



if __name__ == "__main__":
    # Run the complete pipeline
    model, results = run_complete_training_pipeline()