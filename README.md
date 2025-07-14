# MoveNet-Keypoints-Extractor

This project provides a pipeline for extracting human pose keypoints and skeleton images from videos using TensorFlow’s MoveNet, and for training a neural network classifier on the extracted keypoints for exercise recognition or similar tasks.

## Features
- Extracts 17 keypoints (x, y, confidence) per frame from videos using MoveNet Thunder.
- Saves keypoints to CSV files and renders skeleton images as 128x128 grayscale PNGs.
- Trains a dense neural network on the keypoints CSVs for classification.
- Exports the trained model as a TFLite file for mobile/edge deployment.

## Requirements
- Python 3.7–3.10 recommended
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

### 1. Extract Keypoints and Skeleton Images
Place your input videos (e.g., MP4 files) in the `dataset/` directory.

Run the keypoint extraction script:
```bash
python movenet_keypoint_exporter.py
```
- Keypoints CSVs will be saved to `train_data/` (e.g., `Correct_Deadlift_1_keypoints.csv`).
- Skeleton images will be saved to `train_data/images/`.

### 2. Train a Classifier on Keypoints
The classifier uses all `*_keypoints.csv` files in `train_data/`.
- The label for each frame is parsed from the filename (by default, the prefix before the first underscore, e.g., `Correct` in `Correct_Deadlift_1_keypoints.csv`).
- **If you want to classify by exercise type (e.g., `Deadlift`), modify the label extraction logic in `cnn_model.py` accordingly.**

Run the training script:
```bash
python cnn_model.py
```
- Trains a dense neural network on the keypoints.
- Exports the model as `keypoints_dense_model.tflite` with label metadata.

## Notes
- Ensure you have at least two different classes (labels) in your dataset for classification to work.
- The scripts do not use the skeleton images for training; only the keypoints CSVs are used.
- You can adjust the label extraction logic in `cnn_model.py` to suit your dataset naming convention.

## File Structure
- `dataset/` — Place your input videos here.
- `train_data/` — Output keypoints CSVs and images will be saved here.
- `train_data/images/` — Skeleton images for each processed frame.
- `movenet_keypoint_exporter.py` — Extracts keypoints and renders skeletons from videos.
- `cnn_model.py` — Trains a classifier on keypoints and exports a TFLite model.
- `requirements.txt` — Python dependencies.

## Example Workflow
1. Add videos to `dataset/`.
2. Run `python movenet_keypoint_exporter.py` to extract keypoints and images.
3. Run `python cnn_model.py` to train and export the classifier.

---
For further customization or troubleshooting, see comments in the scripts or contact the author.
