"""Offline processing for UR5e Octo fine-tuning episodes.

This package is intentionally independent of ROS and of the collector package.
It converts one replay episode into timestamp-aligned transition arrays that a
separate RLDS builder can package without recomputing synchronization/actions.
"""

__version__ = "0.1.0"
