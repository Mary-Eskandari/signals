from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

SCG_RHC_RAW_DIR = DATA_RAW / "scg_rhc"
CHIRON_RAW_DIR = DATA_RAW / "chiron"
