"""Testes unitários para o SoccerBallTracker com PyTorch e Ultralytics."""

import numpy as np
import pytest
import torch
from soccer_ball_detector.detector import SoccerBallTracker


class MockYOLOModel:
    """Mock determinístico para testar a máquina de estados do tracker sem pesos externos."""

    def __init__(self, mode="always_detect"):
        self.mode = mode
        self.call_count = 0

    def to(self, device):
        return self

    def __call__(self, tensor, verbose=False):
        self.call_count += 1
        b, c, h, w = tensor.shape

        class MockBox:
            def __init__(self, xywh, conf=0.92, cls_id=0):
                self.xywh = [torch.tensor(xywh)]
                self.conf = torch.tensor([conf])
                self.cls = torch.tensor([cls_id])

        class MockResult:
            def __init__(self, boxes):
                self.boxes = boxes
                self.names = {0: "bola"}

        if self.mode == "always_detect":
            # Retorna uma caixa centralizada no tensor
            return [MockResult([MockBox([w / 2.0, h / 2.0, 16.0, 16.0])])]
        elif self.mode == "fail_on_roi":
            # Falha se for um recorte pequeno (ROI), detecta se for quadro completo
            if h < 200:
                return [MockResult([])]
            return [MockResult([MockBox([w / 2.0, h / 2.0, 16.0, 16.0])])]
        else:
            return [MockResult([])]


def test_tracker_tensor_preparation():
    """Valida conversão PyTorch, normalização para [0, 1] e padding para múltiplo de 32."""
    tracker = SoccerBallTracker(model=MockYOLOModel(), device="cpu")

    # Imagem com dimensões não divisíveis por 32: 105 x 150
    img = np.random.randint(0, 255, (105, 150, 3), dtype=np.uint8)
    tensor = tracker._prepare_tensor(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 4
    b, c, h, w = tensor.shape
    assert b == 1
    assert c == 3
    # 105 arredonda para cima para 128 (múltiplo de 32)
    # 150 arredonda para cima para 160 (múltiplo de 32)
    assert h % 32 == 0
    assert w % 32 == 0
    assert h == 128
    assert w == 160
    assert tensor.max().item() <= 1.0
    assert tensor.min().item() >= 0.0


def test_tracker_roi_and_fallback_flow():
    """Valida transição de estado entre Full-Frame, Dynamic ROI e Fallback."""
    mock = MockYOLOModel(mode="fail_on_roi")
    tracker = SoccerBallTracker(model=mock, device="cpu")

    frame1 = np.ones((400, 600, 3), dtype=np.uint8) * 50

    # 1. Primeiro quadro: sem histórico, obrigatoriamente executa detecção completa
    bbox1 = tracker.process_frame(frame1, frame_idx=0)
    assert bbox1 is not None
    assert tracker.last_mode == "full"
    assert len(tracker.trajectory_history) == 1

    # 2. Segundo quadro: com last_bbox, tenta ROI primeiro; o mock fail_on_roi simula perda na ROI e aciona fallback
    bbox2 = tracker.process_frame(frame1, frame_idx=1)
    assert bbox2 is not None
    assert tracker.last_mode == "full"  # Acionou fallback
    assert len(tracker.trajectory_history) == 2


def test_tracker_successful_roi():
    """Valida detecção bem-sucedida dentro da ROI dinâmica."""
    mock = MockYOLOModel(mode="always_detect")
    tracker = SoccerBallTracker(model=mock, device="cpu")

    frame = np.ones((400, 600, 3), dtype=np.uint8) * 100

    # Quadro 1: Inicializa no frame completo
    tracker.process_frame(frame, frame_idx=0)
    assert tracker.last_mode == "full"

    # Quadro 2: Deve detectar com sucesso na ROI dinâmica
    bbox = tracker.process_frame(frame, frame_idx=1)
    assert bbox is not None
    assert tracker.last_mode == "roi"

    # Valida histórico de trajetória
    trajectory = tracker.get_trajectory()
    assert len(trajectory) == 2
    assert isinstance(trajectory[0], tuple)


def test_tracker_annotation_rendering():
    """Valida renderização de anotações visuais no frame."""
    tracker = SoccerBallTracker(model=MockYOLOModel(), device="cpu")
    frame = np.zeros((300, 400, 3), dtype=np.uint8)

    tracker.process_frame(frame)
    annotated = tracker.annotate_frame(frame, draw_trajectory=True, show_roi=True)

    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    # Deve conter pixels desenhados diferentes de zero
    assert np.any(annotated != 0)
