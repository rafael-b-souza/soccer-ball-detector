"""Módulo rastreador de bola com arquitetura YOLO e estágio duplo (ROI dinâmica + fallback).

Motivação Acadêmica e de Engenharia:
Em transmissões de partidas de futebol, a velocidade linear da bola pode atingir mais de 100 km/h,
provocando deslocamentos inter-quadros significativos e frequentes desfoques (motion blur).
Ao mesmo tempo, processar cada quadro em resolução completa (1280x720 ou 1920x1080) sobrecarrega a GPU,
reduzindo a taxa de quadros (FPS) para níveis incompatíveis com análise em tempo real.

O SoccerBallTracker emprega uma máquina de estados de rastreamento com dois estágios:
1. Estágio Primário (ROI Dinâmica Otimizada por AG):
   Quando a bola foi detectada no quadro anterior, recorta uma janela centrada nas coordenadas
   conhecidas com escala roi_scale (≈ 4.27x), aplicando a cadeia de pré-processamento otimizada (Gamma 1.21).
   Isso reduz em até 70% a área analisada pelo YOLO11.
2. Estágio Secundário (Fallback para Quadro Completo):
   Se a inferência na ROI não atingir o limiar de confiança (devido a aceleração brusca, oclusão momentânea
   ou saída da janela), o rastreador reexecuta a busca no quadro completo para re-adquirir o alvo.
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from .preproc import apply_pipeline
from .roi import extract_ball_box, get_roi_from_bbox

logger = logging.getLogger(__name__)


class SoccerBallTracker:
    """Rastreador de bola com YOLO e estágio duplo (ROI adaptativa + fallback)."""

    def __init__(
        self,
        model_path: str = "yolov11-small.pt",
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        model: Optional[Any] = None,
    ):
        """Inicializa o rastreador com os pesos do YOLO e hiperparâmetros ótimos do AG.

        Args:
            model_path: Caminho ou nome do arquivo de pesos YOLO (ex: 'yolo11n.pt', 'yolov11-small.pt').
            config_path: Caminho para o JSON de configuração do AG (ex: 'configs/best_preproc_params.json').
            device: Dispositivo de execução ('cuda', 'cpu' ou None para auto-detecção).
            model: Instância pré-carregada do modelo YOLO (opcional, útil para testes e mocks).
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model is not None:
            self.model = model
        else:
            logger.info("Carregando modelo YOLO a partir de %s no dispositivo %s", model_path, self.device)
            self.model = YOLO(model_path).to(self.device)

        # Hiperparâmetros padrão (convergência do Algoritmo Genético documentada no artigo)
        self.params: Dict[str, Any] = {
            "preproc": "gamma",
            "remove_bg": True,
            "lower_blue": 106,
            "upper_blue": 156,
            "clahe_clip": 2.8,
            "tile_size": (16, 16),
            "gamma_value": 1.21,
            "roi_scale": 4.27,
        }

        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                self.params.update(loaded)
                logger.info("Parâmetros do AG carregados de %s: %s", config_path, self.params)

        self.last_bbox: Optional[Tuple[float, float, float, float]] = None
        self.last_mode: str = "none"
        self.last_roi: Optional[Dict[str, int]] = None
        self.trajectory_history: List[Dict[str, Any]] = []

    def _prepare_tensor(self, img_bgr: np.ndarray) -> torch.Tensor:
        """Converte a imagem BGR para tensor PyTorch com dimensões normalizadas e padding múltiplo de 32.

        Regra de Engenharia:
        A arquitetura convolucional com downsampling stride de 32 no YOLO requer que as dimensões
        de entrada (H, W) sejam estritamente divisíveis por 32 para evitar artefatos de interpolação.
        """
        tensor = torch.tensor(img_bgr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        _, _, h, w = tensor.shape
        pad_h = (math.ceil(h / 32) * 32) - h
        pad_w = (math.ceil(w / 32) * 32) - w
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)
        return tensor

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: Optional[int] = None,
        conf_threshold: float = 0.3,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Processa um quadro de vídeo retornando as coordenadas absolutas (cx, cy, w, h) da bola.

        Regra de Negócio:
        Executa primeiro a busca localizada na ROI dinâmica. Se encontrar detecção válida,
        converte as coordenadas relativas da ROI para o espaço absoluto do quadro e atualiza
        a trajetória. Caso contrário, aciona o fallback no quadro completo.

        Args:
            frame: Imagem BGR (numpy.ndarray).
            frame_idx: Índice sequencial do quadro (opcional, para telemetria da trajetória).
            conf_threshold: Limiar mínimo de confiança da predição.

        Returns:
            Tupla (cx, cy, w, h) no espaço do quadro completo ou None se não detectado.
        """
        p = self.params
        use_hist_eq = (p.get("preproc") == "hist_eq")
        use_clahe = (p.get("preproc") == "clahe")
        use_gamma = (p.get("preproc") == "gamma")
        use_remove_bg = bool(p.get("remove_bg", False))

        clahe_clip = float(p.get("clahe_clip", 2.0))
        tile_size = tuple(p.get("tile_size", (8, 8)))
        gamma_value = float(p.get("gamma_value", 1.0))
        roi_scale = float(p.get("roi_scale", 3.0))

        lower_bg = (int(p.get("lower_blue", 100)), 100, 100)
        upper_bg = (int(p.get("upper_blue", 140)), 255, 255)

        roi_coords = None
        box_detected = None
        detection_mode = "none"

        # 1. Tentativa na ROI dinâmica
        if self.last_bbox is not None:
            roi_coords = get_roi_from_bbox(self.last_bbox, frame.shape, roi_scale)
            self.last_roi = roi_coords
            crop = frame[roi_coords["top"]:roi_coords["bottom"], roi_coords["left"]:roi_coords["right"]]

            if crop.shape[0] > 10 and crop.shape[1] > 10:
                preproc_crop = apply_pipeline(
                    crop,
                    use_hist_eq=use_hist_eq,
                    use_clahe=use_clahe,
                    clahe_clip=clahe_clip,
                    tile_size=tile_size,
                    use_gamma=use_gamma,
                    gamma_value=gamma_value,
                    use_remove_bg=use_remove_bg,
                    lower_bg=lower_bg,
                    upper_bg=upper_bg,
                )
                crop_tensor = self._prepare_tensor(preproc_crop)
                results = self.model(crop_tensor, verbose=False)
                box = extract_ball_box(results, preproc_crop, threshold=conf_threshold)

                if box is not None:
                    cx, cy, bw, bh = box
                    # Translação para coordenadas globais
                    box_detected = (cx + roi_coords["left"], cy + roi_coords["top"], bw, bh)
                    detection_mode = "roi"

        # 2. Fallback: inferência no quadro completo
        if box_detected is None:
            preproc_full = apply_pipeline(
                frame,
                use_hist_eq=use_hist_eq,
                use_clahe=use_clahe,
                clahe_clip=clahe_clip,
                tile_size=tile_size,
                use_gamma=use_gamma,
                gamma_value=gamma_value,
                use_remove_bg=use_remove_bg,
                lower_bg=lower_bg,
                upper_bg=upper_bg,
            )
            full_tensor = self._prepare_tensor(preproc_full)
            results = self.model(full_tensor, verbose=False)
            box = extract_ball_box(results, preproc_full, threshold=conf_threshold)

            if box is not None:
                box_detected = box
                detection_mode = "full"
            else:
                detection_mode = "lost"

        self.last_bbox = box_detected
        self.last_mode = detection_mode

        # Registro na trajetória
        record = {
            "frame_idx": frame_idx if frame_idx is not None else len(self.trajectory_history),
            "bbox": box_detected,
            "center": (box_detected[0], box_detected[1]) if box_detected is not None else None,
            "mode": detection_mode,
            "roi": roi_coords,
        }
        self.trajectory_history.append(record)

        return self.last_bbox

    def get_trajectory(self) -> List[Tuple[float, float]]:
        """Retorna lista sequencial de centroides (cx, cy) detectados ao longo do histórico."""
        return [r["center"] for r in self.trajectory_history if r["center"] is not None]

    def clear_trajectory(self) -> None:
        """Limpa o histórico de rastreamento e redefine o estado."""
        self.trajectory_history.clear()
        self.last_bbox = None
        self.last_mode = "none"
        self.last_roi = None

    def annotate_frame(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        draw_trajectory: bool = True,
        max_trail: int = 30,
        show_roi: bool = True,
    ) -> np.ndarray:
        """Renderiza anotações visuais da bola, rastro da trajetória e janela ROI dinâmica.

        Args:
            frame: Imagem BGR sobre a qual desenhar.
            bbox: Bounding box atual (se None, usa self.last_bbox).
            draw_trajectory: Habilita rastro histórico neon da trajetória.
            max_trail: Comprimento máximo em quadros do rastro histórico.
            show_roi: Desenha a caixa delimitadora da ROI ativa.

        Returns:
            Imagem anotada (numpy.ndarray).
        """
        annotated = frame.copy()
        current_bbox = bbox if bbox is not None else self.last_bbox

        # 1. Desenho da ROI dinâmica (se ativa)
        if show_roi and self.last_roi is not None:
            r = self.last_roi
            cv2.rectangle(annotated, (r["left"], r["top"]), (r["right"], r["bottom"]), (255, 200, 0), 1, cv2.LINE_AA)
            cv2.putText(
                annotated,
                f"ROI Dynamic ({self.params.get('roi_scale', 4.27):.1f}x)",
                (r["left"] + 4, max(18, r["top"] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 200, 0),
                1,
                cv2.LINE_AA,
            )

        # 2. Desenho do rastro da trajetória com gradiente de transparência/cor
        if draw_trajectory:
            centers = self.get_trajectory()[-max_trail:]
            for i in range(1, len(centers)):
                pt1 = (int(centers[i - 1][0]), int(centers[i - 1][1]))
                pt2 = (int(centers[i][0]), int(centers[i][1]))
                alpha = i / len(centers)
                thickness = max(1, int(1 + 2 * alpha))
                # Gradiente de cor: de amarelo (início do rastro) a verde neon (posições recentes)
                color = (int(50 * (1 - alpha)), int(255 * alpha), int(255 * (1 - alpha * 0.5)))
                cv2.line(annotated, pt1, pt2, color, thickness, cv2.LINE_AA)
                cv2.circle(annotated, pt2, max(2, int(3 * alpha)), color, -1, cv2.LINE_AA)

        # 3. Desenho da bounding box e centroide da bola
        if current_bbox is not None:
            cx, cy, bw, bh = current_bbox
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)

            box_color = (0, 255, 0) if self.last_mode == "roi" else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
            cv2.circle(annotated, (int(cx), int(cy)), 3, (0, 0, 255), -1, cv2.LINE_AA)

            label = f"Ball [{self.last_mode.upper()}]"
            cv2.putText(
                annotated,
                label,
                (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                1,
                cv2.LINE_AA,
            )

        return annotated
