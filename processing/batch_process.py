"""Process multiple episode directories and write a dataset-level summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .synchronize_episode import synchronize_episode
except ImportError:
    from synchronize_episode import synchronize_episode


def discover_episodes(root: Path) -> list[Path]:
    if (root / "replay" / "robot_states.mcap").is_file():
        return [root]
    return sorted(
        path for path in root.iterdir()
        if path.is_dir() and (path / "replay" / "robot_states.mcap").is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--rate-hz", type=float)
    parser.add_argument("--require-wrist", action="store_true")
    parser.add_argument("--action-frame", choices=("tool", "base"))
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--summary", type=Path)
    arguments = parser.parse_args()

    results = []
    for episode in discover_episodes(arguments.root):
        try:
            results.append(
                synchronize_episode(
                    episode,
                    rate_hz=arguments.rate_hz,
                    require_wrist=arguments.require_wrist,
                    action_frame=arguments.action_frame,
                    thresholds_path=arguments.thresholds,
                )
            )
        except Exception as error:  # report per episode and continue
            results.append(
                {
                    "episode_id": episode.name,
                    "quality": "ERROR",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    summary = {
        "episode_count": len(results),
        "accepted_count": sum(item.get("quality") == "ACCEPTED" for item in results),
        "warning_count": sum(item.get("quality") == "WARN_ACCEPTED" for item in results),
        "rejected_count": sum(item.get("quality") == "REJECT" for item in results),
        "error_count": sum(item.get("quality") == "ERROR" for item in results),
        "episodes": results,
    }
    text = json.dumps(summary, indent=2) + "\n"
    if arguments.summary:
        arguments.summary.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
