# FlexFit ML Pipeline

A streamlined machine learning pipeline for exercise form analysis using pose estimation with MoveNet and TensorFlow.

## 🚀 Quick Start

```bash
# Run the complete pipeline
python run.py

# Or use the advanced pipeline with options
python flexfit_pipeline.py --step extract --force
```

## 📁 Project Structure

```
flexfit_ML_model/
├── run.py                          # Simple runner script
├── flexfit_pipeline.py             # Main pipeline
├── Flexfit_model_design.py         # Model architecture
├── movenet_keypoint_exporter.py    # Keypoint extraction
├── data_pipeline.py                # Data processing
├── pose_training_deployment.py     # Training & deployment
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── dataset/                        # Video dataset
│   ├── correct/                    # Correct form videos
│   └── incorrect/                  # Incorrect form videos
├── keypoints_data/                 # Extracted keypoints
└── __pycache__/                    # Python cache
```

## 🎯 Core Features

- **Keypoint Extraction**: Extract 17 body keypoints from videos using MoveNet
- **CSV Output**: Save keypoints to structured CSV format
- **TFLite Model**: Create deployable TensorFlow Lite model
- **Multi-head Architecture**: Form score + joint alignment detection

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- tensorflow
- numpy
- pandas
- opencv-python
- scikit-learn
- tensorflow-hub

## 🔧 Usage

### Simple Usage
```bash
# Run complete pipeline
python run.py
```

### Advanced Usage
```bash
# Run complete pipeline
python flexfit_pipeline.py

# Run specific step
python flexfit_pipeline.py --step extract
python flexfit_pipeline.py --step train

# Force re-run (ignore existing files)
python flexfit_pipeline.py --force
```

## 📊 Pipeline Steps

### 1. Keypoint Extraction
- Extracts 17 body keypoints from videos using MoveNet
- Processes both correct and incorrect form videos
- Saves keypoints to `keypoints_data/keypoints_dataset.csv`

### 2. Model Training & TFLite Creation
- Trains 1D CNN with multi-head outputs
- Creates TensorFlow Lite model for deployment
- Outputs `pose_model.tflite`

## 🏗️ Model Architecture

The model uses a 1D Convolutional Neural Network with two outputs:

1. **Form Score** (0-1): Overall exercise form quality
2. **Joint Alignment** (17 joints): Binary alignment for each joint

### Loss Functions
- Form Score: Mean Squared Error (MSE)
- Joint Alignment: Binary Cross-Entropy (BCE)

## 🚀 Deployment

### Using TFLite Model
```python
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="pose_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input (17 keypoints, 3 features each)
keypoints = np.random.randn(1, 17, 3).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], keypoints)
interpreter.invoke()

# Get results
form_score = interpreter.get_tensor(output_details[0]['index'])
joint_alignment = interpreter.get_tensor(output_details[1]['index'])
```

## 🔧 Customization

### Adding New Exercises
1. Add videos to `dataset/correct/` and `dataset/incorrect/`
2. Run the pipeline: `python run.py`
3. The model will automatically adapt to new data

### Model Configuration
Edit `Flexfit_model_design.py` to modify:
- Model architecture
- Loss weights
- Training parameters

## 🐛 Troubleshooting

### Common Issues

**Missing Dependencies**
```bash
pip install -r requirements.txt
```

**Dataset Structure**
Ensure your dataset follows this structure:
```
dataset/
├── correct/
│   ├── video1.mp4
│   └── video2.mp4
└── incorrect/
    ├── video3.mp4
    └── video4.mp4
```

**Memory Issues**
- Reduce batch size in training
- Use fewer epochs
- Process videos in smaller batches

## 📝 Logging

The pipeline generates detailed logs:
- `flexfit_pipeline.log`: Complete execution log
- Console output: Real-time progress updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google MoveNet for pose estimation
- TensorFlow for the ML framework
- OpenCV for video processing

---

**FlexFit ML Pipeline** - Streamlined exercise form analysis! 🏋️‍♂️
