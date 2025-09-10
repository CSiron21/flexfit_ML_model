"""
Rep-level model evaluation script for form scoring model.

Metrics:
- Precision
- Recall
- F1-score
- Accuracy

Setup:
- Each test video is stored as a CSV file.
- One CSV = one video.
- CSV has per-frame rows with columns:
    - 'label': int (1 = correct video, 0 = incorrect video)
    - keypoint feature columns (flattened joints)

Rep-level evaluation:
- Each video is assumed to contain 3 repetitions.
- Frames are split evenly into 3 segments (1 segment = 1 rep).
- Model predicts a continuous score [0,1] per frame.
- A rep is classified as incorrect if:
    - Fraction of frames below LOW_SCORE_THRESHOLD ≥ MIN_LOW_FRACTION, OR
    - Longest run of consecutive low scores ≥ MIN_LOW_RUN.
- Otherwise, rep is classified as correct.
- Ground truth label for each rep = same as the video-level label (all reps inherit video label).
- Metrics are computed at the rep level (≈ 3 × number of videos).
- Per-video classification can optionally be derived by majority vote of its reps.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
import logging
from datetime import datetime

# =============================
# CONFIGURATION VARIABLES
# =============================
# Data configuration
EXERCISE = "bicep_curls"  # Change exercise: (overhead_presses, squats, bicep_curls)
DATA_DIR = Path(f"./keypoints_data/{EXERCISE}/test")   # folder containing test CSVs
CSV_GLOB = "*.csv"                                # match pattern for test files

# Model configuration
MODEL_PATH = Path(f"models/{EXERCISE}/{EXERCISE}_float16.tflite")  # path to .tflite model

# Classification thresholds
LOW_SCORE_THRESHOLD = 0.5          # frame below this = low score
MIN_LOW_FRACTION = 0.1             # fraction of frames below threshold → incorrect
MIN_LOW_RUN = 5                     # consecutive low frames → incorrect

# Rep-level configuration
REPS_PER_VIDEO = 3                 # number of repetitions per video

# Logging configuration
LOG_DIR = Path("./inference_logs")               # directory for log files
LOG_FILENAME = f"{EXERCISE}_rep_evaluation.log"  # log file name
LOG_PATH = LOG_DIR / LOG_FILENAME                 # full path to log file


# =============================
# LOGGING SETUP
# =============================
def setup_logging(log_path: Path):
    """Setup logging configuration to save evaluation results to file."""
    # Create log directory if it doesn't exist
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),  # 'w' mode overwrites previous logs
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    # Log evaluation start
    logging.info("=" * 60)
    logging.info("REP-LEVEL FORM SCORING MODEL EVALUATION STARTED")
    logging.info("=" * 60)
    logging.info(f"Evaluation timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Model path: {MODEL_PATH}")
    logging.info(f"Data directory: {DATA_DIR}")
    logging.info(f"Reps per video: {REPS_PER_VIDEO}")
    logging.info(f"Low score threshold: {LOW_SCORE_THRESHOLD}")
    logging.info(f"Min low fraction: {MIN_LOW_FRACTION}")
    logging.info(f"Min low run: {MIN_LOW_RUN}")
    logging.info("=" * 60)


# =============================
# MODEL PREDICTION
# =============================
def model_predict(interpreter: tf.lite.Interpreter,
                  input_details,
                  output_details,
                  X: np.ndarray) -> np.ndarray:
    """
    Run inference on input X using TFLite interpreter.
    Returns form_score predictions per frame.
    """
    scores = []
    for frame in X:
        frame = frame.astype(np.float32).reshape(1, -1)  # [1, n_features]
        interpreter.set_tensor(input_details[0]['index'], frame)
        interpreter.invoke()
        # Get form_score from output 1 (not output 0)
        form_score = interpreter.get_tensor(output_details[1]['index'])  # (1,1)
        scores.append(form_score.item())
    return np.array(scores, dtype=np.float32)


# =============================
# UTILITIES
# =============================
def max_consecutive_true(mask: np.ndarray) -> int:
    max_run = run = 0
    for val in mask:
        if val:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def classify_rep(scores: np.ndarray, low_threshold: float,
                 min_low_fraction: float, min_low_run: int) -> int:
    """Return predicted label for a rep (1=correct, 0=incorrect)."""
    low_mask = scores < low_threshold
    low_fraction = np.mean(low_mask)
    longest_run = max_consecutive_true(low_mask)

    if low_fraction >= min_low_fraction or longest_run >= min_low_run:
        return 0  # incorrect
    return 1      # correct


def split_video_into_reps(features: np.ndarray, n_reps: int = 3) -> list:
    """Split video frames into n_reps segments."""
    n_frames = len(features)
    frames_per_rep = n_frames // n_reps
    
    reps = []
    for i in range(n_reps):
        start_idx = i * frames_per_rep
        end_idx = start_idx + frames_per_rep if i < n_reps - 1 else n_frames
        reps.append(features[start_idx:end_idx])
    
    return reps





# =============================
# MAIN EVALUATION
# =============================
def evaluate_reps(data_dir: Path, csv_glob: str, low_threshold: float, 
                  min_low_fraction: float, min_low_run: int, n_reps: int,
                  interpreter, input_details, output_details):
    rep_results = []

    for csv_file in data_dir.glob(csv_glob):
        df = pd.read_csv(csv_file)
        y_true_video = int(df['label'].iloc[0])  # video-level label
        
        # Extract features (exclude label column)
        features = df.drop(columns=['label']).values
        
        # Split video into reps
        rep_features = split_video_into_reps(features, n_reps)
        
        # Evaluate each rep
        rep_predictions = []
        rep_scores = []
        
        for rep_idx, rep_feat in enumerate(rep_features):
            # Run inference to get scores for this rep
            scores = model_predict(interpreter, input_details, output_details, rep_feat)
            
            # Classify rep based on scores
            y_pred_rep = classify_rep(scores, low_threshold, min_low_fraction, min_low_run)
            
            rep_predictions.append(y_pred_rep)
            rep_scores.append(scores)
            
            # Store rep-level results
            rep_results.append({
                "video": csv_file.name,
                "rep": rep_idx + 1,
                "true": y_true_video,  # rep inherits video label
                "pred": y_pred_rep,
                "mean_score": np.mean(scores),
                "min_score": np.min(scores),
                "low_fraction": np.mean(scores < low_threshold),
                "n_frames": len(scores)
            })
        


    rep_results_df = pd.DataFrame(rep_results)
    
    # Compute rep-level metrics
    y_true_reps = rep_results_df['true']
    y_pred_reps = rep_results_df['pred']

    rep_precision = precision_score(y_true_reps, y_pred_reps, zero_division=0)
    rep_recall = recall_score(y_true_reps, y_pred_reps, zero_division=0)
    rep_f1 = f1_score(y_true_reps, y_pred_reps, zero_division=0)
    rep_accuracy = accuracy_score(y_true_reps, y_pred_reps)

    # Log rep-level results
    logging.info("=== Rep-level results ===")
    for _, row in rep_results_df.iterrows():
        logging.info(f"{row['video']} - Rep {row['rep']}: true={row['true']}, pred={row['pred']}, "
                    f"mean_score={row['mean_score']:.3f}, min_score={row['min_score']:.3f}, "
                    f"low_fraction={row['low_fraction']:.3f}, frames={row['n_frames']}")
    
    # Log rep-level metrics
    logging.info("=== Rep-level metrics ===")
    logging.info(f"Precision: {rep_precision:.3f}")
    logging.info(f"Recall:    {rep_recall:.3f}")
    logging.info(f"F1-score:  {rep_f1:.3f}")
    logging.info(f"Accuracy:  {rep_accuracy:.3f}")
    logging.info(f"Total reps evaluated: {len(rep_results_df)}")

    # Also print to console for immediate feedback
    print("=== Rep-level results ===")
    print(rep_results_df)
    print("\n=== Rep-level metrics ===")
    print(f"Precision: {rep_precision:.3f}")
    print(f"Recall:    {rep_recall:.3f}")
    print(f"F1-score:  {rep_f1:.3f}")
    print(f"Accuracy:  {rep_accuracy:.3f}")
    print(f"Total reps evaluated: {len(rep_results_df)}")

    return (rep_results_df, 
            {"precision": rep_precision, "recall": rep_recall, "f1": rep_f1, "accuracy": rep_accuracy})


if __name__ == "__main__":
    # =============================
    # SETUP LOGGING
    # =============================
    setup_logging(LOG_PATH)
    
    # =============================
    # LOAD MODEL
    # =============================
    logging.info("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.info("Model loaded successfully.")

    # =============================
    # RUN EVALUATION
    # =============================
    logging.info("Starting rep-level evaluation...")
    results = evaluate_reps(DATA_DIR, CSV_GLOB, LOW_SCORE_THRESHOLD, MIN_LOW_FRACTION, MIN_LOW_RUN,
                           REPS_PER_VIDEO, interpreter, input_details, output_details)
    
    rep_results_df, metrics = results
    
    # Log completion
    logging.info("=" * 60)
    logging.info("REP-LEVEL EVALUATION COMPLETED SUCCESSFULLY")
    logging.info("=" * 60)
    logging.info(f"Results saved to: {LOG_PATH}")
    logging.info(f"Total reps evaluated: {len(rep_results_df)}")
    logging.info(f"Rep-level accuracy: {metrics['accuracy']:.3f}")
    logging.info("=" * 60)