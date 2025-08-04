import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class PoseAnalysisCNN(tf.keras.Model):
    """
    1D CNN model for pose analysis with multi-head outputs:
    - Form score regression (0-1)
    - Per-joint alignment detection (17 joints)
    - Joint angles auxiliary head
    """
    
    def __init__(self, num_joints=17, num_features=3, **kwargs):
        super(PoseAnalysisCNN, self).__init__(**kwargs)
        
        self.num_joints = num_joints
        self.num_features = num_features
        
        # Shared CNN backbone
        self.conv1 = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.2)
        
        self.conv2 = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.2)
        
        self.conv3 = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')
        self.bn3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(0.3)
        
        # Global pooling and dense layers
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense_shared = layers.Dense(512, activation='relu')
        self.dropout_shared = layers.Dropout(0.4)
        
        # Head 1: Form score (regression, 0-1)
        self.form_dense1 = layers.Dense(256, activation='relu')
        self.form_dense2 = layers.Dense(128, activation='relu')
        self.form_output = layers.Dense(1, activation='sigmoid', name='form_score')
        
        # Head 2: Per-joint alignment (sigmoid outputs for 17 joints)
        self.alignment_dense1 = layers.Dense(256, activation='relu')
        self.alignment_dense2 = layers.Dense(128, activation='relu')  
        self.alignment_output = layers.Dense(num_joints, activation='sigmoid', name='joint_alignment')
        
        # Auxiliary Head: Joint angles
        self.angles_dense1 = layers.Dense(256, activation='relu')
        self.angles_dense2 = layers.Dense(128, activation='relu')
        # Assuming we want to predict angles for each joint (17 angles)
        self.angles_output = layers.Dense(num_joints, activation='linear', name='joint_angles')
    
    def call(self, inputs, training=None):
        # Input shape: (batch_size, 17, 3)
        
        # Shared CNN backbone
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.dropout3(x, training=training)
        
        # Global pooling and shared dense
        x = self.global_pool(x)
        x = self.dense_shared(x)
        shared_features = self.dropout_shared(x, training=training)
        
        # Head 1: Form score
        form_x = self.form_dense1(shared_features)
        form_x = self.form_dense2(form_x)
        form_score = self.form_output(form_x)
        
        # Head 2: Joint alignment
        align_x = self.alignment_dense1(shared_features)
        align_x = self.alignment_dense2(align_x)
        joint_alignment = self.alignment_output(align_x)
        
        # Auxiliary Head: Joint angles
        angles_x = self.angles_dense1(shared_features)
        angles_x = self.angles_dense2(angles_x)
        joint_angles = self.angles_output(angles_x)
        
        return {
            'form_score': form_score,
            'joint_alignment': joint_alignment,
            'joint_angles': joint_angles
        }

def create_pose_model(num_joints=17, num_features=3):
    """
    Create and compile the pose analysis model
    """
    model = PoseAnalysisCNN(num_joints=num_joints, num_features=num_features)
    
    # Define loss functions
    losses = {
        'form_score': 'mse',  # MSE for form score regression
        'joint_alignment': 'binary_crossentropy',  # BCE for per-joint misalignment
        'joint_angles': 'mse'  # MSE for joint angles
    }
    
    # Set loss weights (form: 0.5, alignment: 1.0, angles: 0.3)
    loss_weights = {
        'form_score': 0.5,
        'joint_alignment': 1.0,
        'joint_angles': 0.3
    }
    
    # Define metrics
    metrics = {
        'form_score': ['mae'],
        'joint_alignment': ['binary_accuracy'],
        'joint_angles': ['mae']
    }
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    return model

def generate_sample_data(batch_size=32, num_joints=17, num_features=3):
    """
    Generate sample data for testing the model
    """
    # Input: pose keypoints (batch_size, 17, 3)
    X = np.random.randn(batch_size, num_joints, num_features).astype(np.float32)
    
    # Outputs
    y = {
        'form_score': np.random.uniform(0, 1, (batch_size, 1)).astype(np.float32),
        'joint_alignment': np.random.randint(0, 2, (batch_size, num_joints)).astype(np.float32),
        'joint_angles': np.random.uniform(-180, 180, (batch_size, num_joints)).astype(np.float32)
    }
    
    return X, y

# Example usage
if __name__ == "__main__":
    # Create model
    model = create_pose_model()
    
    # Build model with sample input
    sample_input = tf.random.normal((1, 17, 3))
    _ = model(sample_input)
    
    # Print model summary
    print("Model Architecture:")
    model.summary()
    
    # Generate sample data
    X_train, y_train = generate_sample_data(batch_size=100)
    X_val, y_val = generate_sample_data(batch_size=20)
    
    print(f"\nInput shape: {X_train.shape}")
    print(f"Form score shape: {y_train['form_score'].shape}")
    print(f"Joint alignment shape: {y_train['joint_alignment'].shape}")
    print(f"Joint angles shape: {y_train['joint_angles'].shape}")
    
    # Train model (example)
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=16,
        verbose=1
    )
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_val[:5])
    
    print("Sample predictions:")
    for i in range(3):
        print(f"Sample {i+1}:")
        print(f"  Form Score: {predictions['form_score'][i][0]:.3f}")
        print(f"  Joint Alignment (first 5 joints): {predictions['joint_alignment'][i][:5]}")
        print(f"  Joint Angles (first 5 joints): {predictions['joint_angles'][i][:5]}")
        print()

# Additional utility functions
class PoseDataProcessor:
    """
    Utility class for preprocessing pose data
    """
    
    def __init__(self, normalize=True):
        self.normalize = normalize
        self.mean = None
        self.std = None
    
    def fit(self, X):
        """Fit preprocessing parameters"""
        if self.normalize:
            self.mean = np.mean(X, axis=(0, 1), keepdims=True)
            self.std = np.std(X, axis=(0, 1), keepdims=True)
    
    def transform(self, X):
        """Transform input data"""
        if self.normalize and self.mean is not None:
            X = (X - self.mean) / (self.std + 1e-8)
        return X
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)

def evaluate_pose_model(model, X_test, y_test):
    """
    Comprehensive evaluation of the pose model
    """
    # Get predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    form_mae = np.mean(np.abs(predictions['form_score'] - y_test['form_score']))
    
    # For joint alignment (binary classification)
    alignment_preds = (predictions['joint_alignment'] > 0.5).astype(int)
    alignment_acc = np.mean(alignment_preds == y_test['joint_alignment'])
    
    # For joint angles
    angles_mae = np.mean(np.abs(predictions['joint_angles'] - y_test['joint_angles']))
    
    print("Model Evaluation Results:")
    print(f"Form Score MAE: {form_mae:.4f}")
    print(f"Joint Alignment Accuracy: {alignment_acc:.4f}")
    print(f"Joint Angles MAE: {angles_mae:.4f}")
    
    return {
        'form_mae': form_mae,
        'alignment_accuracy': alignment_acc,
        'angles_mae': angles_mae
    }