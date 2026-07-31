"""Compatibility wrapper for the installed collector synchronization module."""

from octo_ur5e_collector.core.synchronize_episode import *  # noqa: F401,F403
from octo_ur5e_collector.core.synchronize_episode import main


if __name__ == "__main__":
    main()
