"""Selected-frame video decode boundary for a future RLDS builder.

This module deliberately does not build RLDS/TFRecord files. OpenCV is imported
only when decoding is requested, so timestamp/state processing remains usable
without it.
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable


def decode_selected_rgb(video_path: str | Path, indices: Iterable[int]):
    import cv2

    requested=[int(value) for value in indices]
    if any(value<0 for value in requested):
        raise ValueError("frame indices must be nonnegative")
    wanted=sorted(set(requested));decoded={}
    capture=cv2.VideoCapture(str(video_path))
    if not capture.isOpened():raise RuntimeError(f"cannot open video: {video_path}")
    frame_index=0;target_index=0
    try:
        while target_index<len(wanted):
            ok,bgr=capture.read()
            if not ok:raise RuntimeError(f"video ended before selected frame {wanted[target_index]}")
            if frame_index==wanted[target_index]:
                decoded[frame_index]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                target_index+=1
            frame_index+=1
    finally:
        capture.release()
    return [decoded[index] for index in requested]
