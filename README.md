# FlexFit ML Model

A machine learning pipeline for exercise form analysis using pose estimation with MoveNet and TensorFlow. The system analyzes exercise form quality and provides real-time feedback for squats, bicep curls, and overhead presses.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model (default: squats)
python train.py

# Extract keypoints from videos
python movenet_keypoint_exporter.py

# Augment existing keypoint data
python augment_data.py
```

## 📁 Project Structure

```
flexfit_ML_model/
├── cnns/                           # CNN model architectures
│   ├── base_pose_cnn.py           # Base CNN class with common functionality
│   ├── squats_cnn.py              # Squat-specific CNN with biomechanical features
│   ├── bicep_curls_cnn.py         # Bicep curl-specific CNN
│   └── overhead_presses_cnn.py    # Overhead press-specific CNN
├── keypoints_data/                 # Extracted pose keypoints
│   ├── squats/
│   │   ├── correct/               # Correct form keypoints
│   │   ├── incorrect/             # Incorrect form keypoints
│   │   └── metadata/              # Keypoint metadata
│   ├── bicep_curls/
│   └── overhead_presses/
├── models/                         # Trained models
│   ├── squats/
│   │   ├── squat_pose_model.keras
│   │   └── squat_pose_float16.tflite
│   ├── bicep_curls/
│   └── overhead_presses/
├── videos_dataset/                 # Original video files
├── training_logs/                  # Training progress logs
├── movenet_keypoint_exporter.py   # Extract keypoints from videos
├── augment_data.py                # Data augmentation utilities
├── train.py                       # Model training script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🎯 Core Features

- **Multi-Exercise Support**: Squats, bicep curls, and overhead presses
- **Biomechanical Analysis**: Advanced joint angle calculations and form detection
- **TFLite Optimization**: Models optimized for mobile deployment
- **Real-time Feedback**: Form scoring and correction instructions
- **Data Augmentation**: Mirroring and jittering for improved training

## 📋 Requirements

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `tensorflow==2.20.0` - Deep learning framework
- `opencv-python==4.12.0.88` - Video processing
- `numpy==2.2.4` - Numerical computations
- `pandas==2.3.1` - Data manipulation
- `scikit-learn==1.7.1` - Machine learning utilities

## 🔧 Usage

### Training Models

```bash
# Train squats model (default)
python train.py

# Train other exercises by modifying EXERCISE variable in train.py:
# EXERCISE = "bicep_curls" or "overhead_presses"
```

### Keypoint Extraction

```bash
# Extract keypoints from videos
python movenet_keypoint_exporter.py

# Configure exercise and label in the script:
# EXERCISE = "squats" | "bicep_curls" | "overhead_presses"
# LABEL = "correct" | "incorrect"
```

### Data Augmentation

```bash
# Augment existing keypoint data
python augment_data.py

# Supports jittering and mirroring for data expansion
```

## 🏗️ Model Architecture

### CNN Architecture
Each exercise has a specialized 1D CNN with:
- **Input**: 17 body keypoints (y, x, confidence) flattened to 51 features
- **CNN Backbone**: 3 convolutional layers with batch normalization
- **Biomechanical Features**: Engineered features for joint angles and form analysis
- **Multi-head Output**: Form score + instruction ID + joint masks

### Key Features
- **TFLite Compatible**: All operations optimized for mobile deployment
- **Biomechanical Analysis**: Advanced joint angle calculations using polynomial approximations
- **Orientation-Aware**: Detects user facing direction for accurate form assessment
- **Confidence Handling**: Robust side selection based on joint confidence scores

## 📊 Data Structure

### Keypoints Data
- **Format**: CSV files with 52 columns (label + 51 keypoint features)
- **Organization**: Separated by exercise type and form quality
- **Augmentation**: Supports mirroring and jittering for data expansion

### Models Output
- **Keras Models**: Full precision models for training and evaluation
- **TFLite Models**: Optimized for mobile deployment with float16 quantization
- **Metadata**: Training logs and model performance metrics

## 🚀 Deployment

### Using TFLite Models
```python
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="models/squats/squat_pose_float16.tflite")
interpreter.allocate_tensors()

# Prepare input (51 keypoint features)
keypoints = np.random.randn(1, 51).astype(np.float32)

# Run inference
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], keypoints)
interpreter.invoke()

# Get results
form_score = interpreter.get_tensor(output_details[0]['index'])
instruction_id = interpreter.get_tensor(output_details[1]['index'])
joint_masks = interpreter.get_tensor(output_details[2]['index'])
```

## 🔧 Customization

### Adding New Exercises
1. Create exercise-specific CNN in `cnns/` directory
2. Add video data to `videos_dataset/`
3. Extract keypoints using `movenet_keypoint_exporter.py`
4. Train model using `train.py`

### Model Configuration
- Modify hyperparameters in `train.py`
- Adjust biomechanical thresholds in CNN files
- Customize data augmentation in `augment_data.py`

## 🐛 Troubleshooting

### Common Issues

**GPU Memory Issues**
```bash
# Reduce batch size in train.py
BATCH_SIZE = 16  # Instead of 32
```

**Missing Dependencies**
```bash
pip install -r requirements.txt
```

**TFLite Conversion Issues**
- Ensure all operations are TFLite-compatible
- Check for custom ops in model architecture

## 📝 Logging

- **Training Logs**: `training_logs/{exercise}_training.log`
- **Console Output**: Real-time training progress
- **Model Checkpoints**: Saved during training for recovery

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Google MoveNet for pose estimation
- TensorFlow for the ML framework
- OpenCV for video processing

---

**FlexFit ML Model** - Advanced exercise form analysis with biomechanical insights! 🏋️‍♂️
