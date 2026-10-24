"""Testes unitários para o módulo de pré-processamento de imagens."""

import numpy as np
import pytest
from soccer_ball_detector.preproc import (
    apply_pipeline,
    clahe,
    gamma_correction,
    hist_eq,
    remove_bg,
)


def test_preprocessing_transforms():
    """Valida as transformações fotométricas básicas e manutenção de dimensionalidade."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    out_hist = hist_eq(img)
    assert out_hist.shape == img.shape
    assert out_hist.dtype == np.uint8

    out_clahe = clahe(img, clip_limit=2.0, tile_grid_size=(4, 4))
    assert out_clahe.shape == img.shape
    assert out_clahe.dtype == np.uint8

    out_gamma = gamma_correction(img, gamma=1.2)
    assert out_gamma.shape == img.shape
    assert out_gamma.dtype == np.uint8

    out_bg = remove_bg(img, lower_bound=(90, 100, 100), upper_bound=(140, 255, 255))
    assert out_bg.shape == img.shape
    assert out_bg.dtype == np.uint8

    composed = apply_pipeline(img, use_clahe=True, use_gamma=True, gamma_value=1.2)
    assert composed.shape == img.shape
    assert composed.dtype == np.uint8


def test_gamma_edge_cases():
    """Testa robustez contra valores extremos ou inválidos de gamma."""
    img = np.ones((20, 20, 3), dtype=np.uint8) * 128

    # Gamma zero ou negativo deve fazer fallback para 1.0 sem lançar exceção
    out_zero = gamma_correction(img, gamma=0.0)
    assert out_zero.shape == img.shape

    out_neg = gamma_correction(img, gamma=-2.5)
    assert out_neg.shape == img.shape


def test_pipeline_all_enabled():
    """Valida execução completa da pipeline encadeando todas as etapas ótimas do AG."""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    out = apply_pipeline(
        img,
        use_hist_eq=True,
        use_clahe=True,
        clahe_clip=2.84,
        tile_size=(16, 16),
        use_gamma=True,
        gamma_value=1.21,
        use_remove_bg=True,
        lower_bg=(106, 100, 100),
        upper_bg=(156, 255, 255),
    )
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_empty_image_handling():
    """Garante tratamento gracioso para imagens vazias ou None."""
    empty = np.array([], dtype=np.uint8)
    assert apply_pipeline(empty).size == 0
    assert apply_pipeline(None) is None
