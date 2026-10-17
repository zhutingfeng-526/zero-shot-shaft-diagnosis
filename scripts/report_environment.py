"""Print current numerical settings without applying a precision profile."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shaft_diagnosis.runtime import runtime_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", action="store_true", help="Also make a read-only nvidia-smi query")
    args = parser.parse_args()
    print(json.dumps(runtime_report(args.driver), indent=2, ensure_ascii=False))
