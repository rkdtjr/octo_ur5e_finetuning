"""Print shapes and sanity statistics from synchronized_episode.npz."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .build_actions import action_statistics
except ImportError:
    from build_actions import action_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_npz", type=Path)
    arguments = parser.parse_args()
    with np.load(arguments.processed_npz, allow_pickle=False) as archive:
        output = {
            "arrays": {
                name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)}
                for name in archive.files
            },
            "valid_ratio": float(archive["valid_mask"].mean()),
            "action_statistics": action_statistics(archive["actions"]),
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
