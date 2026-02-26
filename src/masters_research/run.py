import sys
from pathlib import Path

# Ensure src is on the import path
ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from masters_research.cli.main import main

if __name__ == "__main__":
    main()
