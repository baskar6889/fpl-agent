"""Make the src-layout package importable during local pytest runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
