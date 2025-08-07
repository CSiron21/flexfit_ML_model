#!/usr/bin/env python3
"""
FlexFit ML Pipeline - Simple Runner
==================================

Quick start script for the FlexFit exercise form analysis pipeline.

Usage:
    python run.py                    # Run complete pipeline
"""

import sys
from flexfit_pipeline import FlexFitPipeline

def main():
    """Simple runner for the FlexFit pipeline"""
    print("FlexFit ML Pipeline")
    print("=" * 40)
    
    # Create pipeline
    pipeline = FlexFitPipeline()
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\nPipeline completed successfully!")
        print("Generated files:")
        print("  - keypoints_data/keypoints_dataset.csv")
        print("  - pose_model.tflite")
        print("  - flexfit_pipeline.log")
    else:
        print("\nPipeline failed! Check the log for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
