#!/usr/bin/env python3
"""
FlexFit ML Pipeline - Core Functionality
========================================

Extract keypoints from videos and create TFLite model for exercise form analysis.

Usage:
    python flexfit_pipeline.py                    # Run complete pipeline
    python flexfit_pipeline.py --step extract     # Run only keypoint extraction
    python flexfit_pipeline.py --step train       # Run only model training
    python flexfit_pipeline.py --force           # Force re-run all steps
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flexfit_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FlexFitPipeline:
    """Core FlexFit ML Pipeline - Keypoint extraction and TFLite model creation"""
    
    def __init__(self):
        self.config = {
            'dataset_path': 'dataset',
            'keypoints_dir': 'keypoints_data',
            'model_path': 'best_pose_model.h5',
            'tflite_path': 'pose_model.tflite'
        }
        
    def check_dependencies(self) -> bool:
        """Check if all required packages are installed"""
        logger.info("Checking dependencies...")
        
        required_packages = [
            ('tensorflow', 'tensorflow'),
            ('numpy', 'numpy'),
            ('pandas', 'pandas'),
            ('opencv-python', 'cv2'),
            ('scikit-learn', 'sklearn'),
            ('tensorflow-hub', 'tensorflow_hub')
        ]
        
        missing_packages = []
        for package_name, import_name in required_packages:
            try:
                __import__(import_name)
                logger.info(f"[OK] {package_name}")
            except ImportError:
                logger.error(f"[MISSING] {package_name}")
                missing_packages.append(package_name)
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.error("Install with: pip install " + " ".join(missing_packages))
            return False
        
        logger.info("[OK] All dependencies are installed!")
        return True
    
    def check_dataset_structure(self) -> bool:
        """Verify dataset structure"""
        logger.info("Checking dataset structure...")
        
        required_dirs = [
            f"{self.config['dataset_path']}/correct",
            f"{self.config['dataset_path']}/incorrect"
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                logger.error(f"[ERROR] Missing directory: {dir_path}")
                return False
            
            # Check for video files
            video_files = list(Path(dir_path).glob("*.mp4")) + list(Path(dir_path).glob("*.avi"))
            if not video_files:
                logger.warning(f"[WARNING] No video files found in {dir_path}")
            else:
                logger.info(f"[OK] {dir_path} ({len(video_files)} videos)")
        
        logger.info("[OK] Dataset structure is correct!")
        return True
    
    def step_extract_keypoints(self, force: bool = False) -> bool:
        """Step 1: Extract keypoints from videos to CSV"""
        logger.info("=" * 60)
        logger.info("STEP 1: KEYPOINT EXTRACTION")
        logger.info("=" * 60)
        
        try:
            # Check if keypoints already exist
            if os.path.exists(f"{self.config['keypoints_dir']}/keypoints_dataset.csv") and not force:
                logger.info("[SKIP] Keypoints already extracted. Skipping...")
                return True
            
            # Import and run extraction
            from movenet_keypoint_exporter import process_dataset
            
            logger.info("Extracting keypoints from videos...")
            process_dataset(
                dataset_path=self.config['dataset_path'],
                output_dir=self.config['keypoints_dir'],
                frame_skip=3
            )
            
            logger.info("[OK] Keypoint extraction completed!")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Keypoint extraction failed: {str(e)}")
            return False
    
    def step_train_and_deploy(self, force: bool = False) -> bool:
        """Step 2: Train model and create TFLite"""
        logger.info("=" * 60)
        logger.info("STEP 2: MODEL TRAINING & TFLITE DEPLOYMENT")
        logger.info("=" * 60)
        
        try:
            # Check if TFLite model already exists
            if os.path.exists(self.config['tflite_path']) and not force:
                logger.info("[SKIP] TFLite model already exists. Skipping...")
                return True
            
            # Import and run training pipeline
            from pose_training_deployment import run_complete_training_pipeline
            
            logger.info("Training model and creating TFLite...")
            model, results = run_complete_training_pipeline()
            
            logger.info("[OK] Model training and TFLite deployment completed!")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Model training/deployment failed: {str(e)}")
            return False
    
    def run_complete_pipeline(self, force: bool = False) -> bool:
        """Run the complete pipeline"""
        logger.info("Starting FlexFit ML Pipeline...")
        start_time = time.time()
        
        # Check prerequisites
        if not self.check_dependencies():
            return False
        
        if not self.check_dataset_structure():
            return False
        
        # Run steps
        steps = [
            ('extract', self.step_extract_keypoints),
            ('train_deploy', self.step_train_and_deploy)
        ]
        
        for step_name, step_func in steps:
            if not step_func(force):
                logger.error(f"[ERROR] Pipeline failed at {step_name} step")
                return False
        
        # Summary
        elapsed_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Time: {elapsed_time:.2f} seconds")
        logger.info("Generated files:")
        logger.info(f"  - {self.config['keypoints_dir']}/keypoints_dataset.csv")
        logger.info(f"  - {self.config['tflite_path']}")
        
        return True
    
    def run_single_step(self, step: str, force: bool = False) -> bool:
        """Run a single pipeline step"""
        step_functions = {
            'extract': self.step_extract_keypoints,
            'train': self.step_train_and_deploy
        }
        
        if step not in step_functions:
            logger.error(f"[ERROR] Unknown step: {step}")
            logger.info(f"Available steps: {', '.join(step_functions.keys())}")
            return False
        
        return step_functions[step](force)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='FlexFit ML Pipeline')
    parser.add_argument('--step', choices=['extract', 'train'],
                       help='Run a specific pipeline step')
    parser.add_argument('--force', action='store_true',
                       help='Force re-run steps even if output exists')
    
    args = parser.parse_args()
    
    # Create pipeline instance
    pipeline = FlexFitPipeline()
    
    # Run pipeline
    if args.step:
        success = pipeline.run_single_step(args.step, args.force)
    else:
        success = pipeline.run_complete_pipeline(args.force)
    
    if success:
        logger.info("Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
