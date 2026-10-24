"""Testes unitários para o módulo de ROI e extração de caixas delimitadoras."""

import numpy as np
import pytest
from soccer_ball_detector.roi import extract_ball_box, get_roi_from_bbox


def test_roi_calculation():
    """Valida o cálculo centrado da ROI e multiplicação pela escala."""
    bbox = (50.0, 50.0, 20.0, 20.0)
    roi = get_roi_from_bbox(bbox, (200, 200, 3), roi_scale=3.0)

    # 20 * 3 = 60 de largura e altura. 50 - 30 = 20, 50 + 30 = 80
    assert roi["left"] == 20
    assert roi["top"] == 20
    assert roi["right"] == 80
    assert roi["bottom"] == 80


def test_roi_boundary_clamping():
    """Valida o truncamento de limites quando a bola está próxima às bordas da imagem."""
    # Bola próxima ao canto superior esquerdo (10, 10)
    bbox = (10.0, 10.0, 20.0, 20.0)
    roi = get_roi_from_bbox(bbox, (100, 100, 3), roi_scale=4.27)

    assert roi["left"] == 0
    assert roi["top"] == 0
    assert roi["right"] <= 100
    assert roi["bottom"] <= 100


def test_extract_ball_box_empty():
    """Garante que resultados vazios retornam None sem falha."""
    assert extract_ball_box(None) is None

    class MockEmptyResult:
        boxes = None

    assert extract_ball_box(MockEmptyResult()) is None


def test_extract_ball_box_with_mock():
    """Valida a seleção correta da caixa com maior confiança."""
    class MockBox:
        def __init__(self, xywh, conf, cls_id):
            self.xywh = [np.array(xywh)]
            self.conf = np.array([conf])
            self.cls = np.array([cls_id])

    class MockResult:
        def __init__(self, boxes):
            self.boxes = boxes
            self.names = {0: "bola", 1: "jogador"}

    box_low = MockBox([100, 100, 20, 20], conf=0.45, cls_id=0)
    box_high = MockBox([200, 200, 25, 25], conf=0.92, cls_id=0)
    box_other = MockBox([50, 50, 40, 80], conf=0.98, cls_id=1)

    result = MockResult([box_low, box_high, box_other])
    extracted = extract_ball_box(result, threshold=0.3, target_classes={0})

    assert extracted is not None
    assert extracted == (200.0, 200.0, 25.0, 25.0)
