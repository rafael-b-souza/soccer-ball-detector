"""Testes unitários para o simulador de partidas e compositor visual."""

import numpy as np
import pytest
from soccer_ball_detector.visuals import SyntheticMatchSimulator, VisualDemoComposer


def test_synthetic_match_simulator_determinism():
    """Valida determinismo na geração de quadros e limites da bola."""
    sim = SyntheticMatchSimulator(width=400, height=300, total_frames=20, seed=123)

    frame0, bbox0 = sim.get_frame(0)
    assert frame0.shape == (300, 400, 3)
    assert frame0.dtype == np.uint8
    assert len(bbox0) == 4
    cx, cy, w, h = bbox0
    assert 0 <= cx <= 400
    assert 0 <= cy <= 300

    # Segundo frame no instante central
    frame10, bbox10 = sim.get_frame(10)
    assert frame10.shape == (300, 400, 3)
    # A bola deve ter se movido no eixo X
    assert bbox10[0] > bbox0[0]


def test_visual_demo_composer_output_shape():
    """Valida as dimensões e montagem do dashboard multi-painel."""
    composer = VisualDemoComposer(output_width=640, output_height=360)

    raw = np.zeros((300, 400, 3), dtype=np.uint8)
    ann = raw.copy()
    preproc = raw.copy()
    bbox = (200.0, 150.0, 20.0, 20.0)
    roi = {"left": 150, "top": 100, "right": 250, "bottom": 200}

    dashboard = composer.compose_dashboard_frame(
        raw_frame=raw,
        annotated_frame=ann,
        preproc_frame=preproc,
        tracker_bbox=bbox,
        last_roi=roi,
        frame_idx=1,
        total_frames=10,
    )

    assert dashboard.shape == (360, 640, 3)
    assert dashboard.dtype == np.uint8


def test_visual_demo_comparison_image():
    """Valida a geração da figura comparativa de 4 estágios."""
    composer = VisualDemoComposer()
    frame = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)

    comp = composer.create_pipeline_comparison_image(
        raw_frame=frame,
        gamma_frame=frame,
        bg_removed_frame=frame,
        annotated_frame=frame,
    )

    assert comp.ndim == 3
    assert comp.shape[0] > 700
    assert comp.shape[1] > 1200
