import numpy as np
import pytest
from soccer_ball_detector.preproc import (
    apply_pipeline,
    clahe,
    gamma_correction,
    hist_eq,
    remove_bg,
)
from soccer_ball_detector.roi import get_roi_from_bbox


def test_preprocessing_transforms():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    out_hist = hist_eq(img)
    assert out_hist.shape == img.shape

    out_clahe = clahe(img, clip_limit=2.0, tile_grid_size=(4, 4))
    assert out_clahe.shape == img.shape

    out_gamma = gamma_correction(img, gamma=1.2)
    assert out_gamma.shape == img.shape

    out_bg = remove_bg(img, lower_bound=(90, 100, 100), upper_bound=(140, 255, 255))
    assert out_bg.shape == img.shape

    composed = apply_pipeline(img, use_clahe=True, use_gamma=True, gamma_value=1.2)
    assert composed.shape == img.shape


def test_roi_calculation():
    bbox = (50.0, 50.0, 20.0, 20.0)
    roi = get_roi_from_bbox(bbox, (200, 200, 3), roi_scale=3.0)

    assert roi["left"] == 20
    assert roi["top"] == 20
    assert roi["right"] == 80
    assert roi["bottom"] == 80
