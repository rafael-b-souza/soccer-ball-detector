"""Pipeline de detecção e rastreamento de bola de futebol com YOLO11 e GA."""

from .preproc import apply_pipeline, clahe, gamma_correction, hist_eq, remove_bg
from .roi import extract_ball_box, get_roi_from_bbox

__all__ = [
    "apply_pipeline",
    "clahe",
    "gamma_correction",
    "hist_eq",
    "remove_bg",
    "extract_ball_box",
    "get_roi_from_bbox"
]
