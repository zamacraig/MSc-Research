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

    # A config asking for outer cross-validation is dispatched to the CV driver,
    # which runs the pipeline once per fold and then pools the held-out
    # predictions. Absent (or 1), behaviour is the ordinary single-split run.
    import json
    n_outer = 1
    if config_path is not None and Path(config_path).exists():
        try:
            n_outer = int(json.loads(Path(config_path).read_text()).get("outer_cv_folds", 1))
        except Exception:
            n_outer = 1

    if n_outer > 1:
        from masters_research.pipeline.outer_cv import main as run_outer_cv
        run_outer_cv(base_path=project_root, config_path=config_path)
        print("Outer-CV pipeline finished successfully.")
    else:
        run_training(base_path=project_root, config_path=config_path)
        print("GNN pipeline finished successfully.")


if __name__ == "__main__":
    main()
