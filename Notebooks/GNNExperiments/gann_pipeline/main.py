
import os
import sys
from train import main as run_training

def main():
    """
    Main entry point for the GNN pipeline.
    This script ensures that the working directory is correctly set and then
    initiates the training process.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change the current working directory to the script's directory
    # This helps ensure that relative paths to data files are resolved correctly.
    os.chdir(script_dir)
    
    print("Starting GNN training and evaluation pipeline...")
    
    try:
        run_training()
        print("GNN pipeline finished successfully.")
    except Exception as e:
        print(f"An error occurred during the pipeline execution: {e}")
        # Optionally, add more detailed error logging here
        sys.exit(1)

if __name__ == '__main__':
    main()
