from __future__ import annotations

import os
from pathlib import Path

# Central project data root:
# - override via PROJECT_DATA_DIR
# - default to sibling directory ../Project_Data next to Project_Code
_REPO_DIR = Path(__file__).resolve().parent
_DEFAULT_BASE_DIR = _REPO_DIR.parent / "Project_Data"

BASE_DIR = Path(os.getenv("PROJECT_DATA_DIR", str(_DEFAULT_BASE_DIR))).expanduser().resolve()
DATA_DIR = BASE_DIR / "intermediate"
CACHE_DATA_DIR = BASE_DIR / "cache"
GRAPH_DIR = BASE_DIR / "graphs"
TABLE_DIR = BASE_DIR / "tables"

# Keep parity with notebook setup behavior.
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DATA_DIR.mkdir(parents=True, exist_ok=True)
