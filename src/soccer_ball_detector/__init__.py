"""Pipeline de detecção e rastreamento de bola de futebol com YOLO11 e GA."""

from .detector import SoccerBallTracker
from .optimizer import (
    PREPROC_OPTIONS,
    GeneticOptimizer,
    compute_fitness,
    individual_to_dict,
    mutate_individual,
    safe_crossover,
)
from .preproc import apply_pipeline, clahe, gamma_correction, hist_eq, remove_bg
from .roi import extract_ball_box, get_roi_from_bbox
from .video import annotate_video

__all__ = [
    "SoccerBallTracker",
    "GeneticOptimizer",
    "compute_fitness",
    "individual_to_dict",
    "mutate_individual",
    "safe_crossover",
    "PREPROC_OPTIONS",
    "apply_pipeline",
    "clahe",
    "gamma_correction",
    "hist_eq",
    "remove_bg",
    "extract_ball_box",
    "get_roi_from_bbox",
    "annotate_video",
]
