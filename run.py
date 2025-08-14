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
        # Print results summary after generated files
        try:
            import json
            with open('results_summary.json', 'r') as f:
                summary = json.load(f)
            print("\nResults:")
            print(
                "  Alignment → "
                f"Accuracy: {summary['alignment']['accuracy']:.4f}, "
                f"Precision: {summary['alignment']['precision']:.4f}, "
                f"Recall: {summary['alignment']['recall']:.4f}, "
                f"F1: {summary['alignment']['f1']:.4f}, "
                f"Threshold: {summary['alignment']['optimal_threshold']:.3f}"
            )
            print(
                "  Form Score → "
                f"MAE: {summary['form']['mae']:.4f}, "
                f"MSE: {summary['form']['mse']:.4f}, "
                f"RMSE: {summary['form']['rmse']:.4f}"
            )
            print(
                "  Joint Angles → "
                f"MAE: {summary['angles']['mae']:.4f}, "
                f"MSE: {summary['angles']['mse']:.4f}, "
                f"RMSE: {summary['angles']['rmse']:.4f}"
            )
        except Exception:
            pass
    else:
        print("\nPipeline failed! Check the log for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
