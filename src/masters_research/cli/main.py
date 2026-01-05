import os
from pathlib import Path
from masters_research.pipeline.train import main as run_training


def main():
    """
    CLI entry point for the MastersResearch pipeline.
    Detects the project root and runs training.
    """
    # Determine project root (assumes src/masters_research/cli/main.py location)
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]

    # Set working directory to project root for consistent relative paths
    os.chdir(project_root)
    print(f"Project root: {project_root}")
    print("Starting GNN training and evaluation pipeline...")

    run_training(base_path=project_root)
    print("GNN pipeline finished successfully.")


if __name__ == "__main__":
    main()
