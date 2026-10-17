"""Create a deterministic artificial 9.6 s history for a CLI smoke test."""
import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).absolute().parents[1]


def example_data():
    """Return synchronized sine waves in the documented 10-channel order."""
    time = np.arange(48000, dtype=np.float64) / 5000.0
    phase = np.arange(10, dtype=np.float64)[:, None] * np.pi / 7.0
    amplitude = np.array([.04, .03, .05, .04, .06, .05, .01, .01, .02, .02])[:, None]
    waveform = amplitude * (
        np.sin(2.0 * np.pi * 20.0 * time + phase)
        + .2 * np.sin(2.0 * np.pi * 40.0 * time - phase)
    )
    return {"raw": waveform.astype(np.float32), "rpm": np.array(1200, dtype=np.float32)}


def write_example(path):
    """Write numeric NPZ fields without replacing an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("Output already exists; choose a new filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        np.savez_compressed(output, **example_data())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/example_history.npz")
    args = parser.parse_args()
    try:
        write_example(args.output)
    except FileExistsError as error:
        parser.error(str(error))
    print("Saved " + str(args.output))
    print("Artificial input only: its predicted labels are not experimental results.")


if __name__ == "__main__":
    main()
