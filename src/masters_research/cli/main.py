import os
import argparse
from pathlib import Path
from masters_research.pipeline.train import main as run_training


def main():
    """
    CLI entry point for the MastersResearch pipeline.
    Detects the project root and runs training.
    """
    parser = argparse.ArgumentParser(description="MastersResearch GNN Training Pipeline")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config JSON file (default: configs/default.json)"
    )
    args = parser.parse_args()
    
    # Determine project root (assumes src/masters_research/cli/main.py location)
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]

    # Set working directory to project root for consistent relative paths
    os.chdir(project_root)
    print(f"Project root: {project_root}")
    print("Starting GNN training and evaluation pipeline...")
    
    # Resolve config path
    config_path = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path
        print(f"Using config: {config_path}")

    run_training(base_path=project_root, config_path=config_path)
    print("GNN pipeline finished successfully.")


if __name__ == "__main__":
    main()
