"""Módulo de síntese de partidas e geração de composições visuais do pipeline de detecção.

Motivação Acadêmica e de Engenharia:
A validação de algoritmos de rastreamento de alvos rápidos requer benchmarks visuais reproduzíveis,
capazes de simular condições desafiadoras de jogo: iluminação heterogênea por refletores de estádio,
sombras de jogadores, textura do gramado e trajetórias balísticas da bola com desfoque de movimento.
Este módulo provê um simulador de partidas de alta fidelidade e um compositor de painéis de visualização
lado a lado (dashboard multi-estágio) para demonstração visual e geração de assets acadêmicos (GIFs e PNGs).
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .preproc import apply_pipeline
from .roi import get_roi_from_bbox


class SyntheticMatchSimulator:
    """Gerador determinístico de quadros realistas de partidas de futebol para testes e demonstrações."""

    def __init__(
        self,
        width: int = 840,
        height: int = 560,
        total_frames: int = 60,
        seed: int = 42,
    ):
        """Inicializa o simulador com dimensões do campo e duração do clipe.

        Args:
            width: Largura em pixels do campo sintético.
            height: Altura em pixels do campo sintético.
            total_frames: Quantidade total de quadros na sequência (padrão: 60).
            seed: Semente pseudo-aleatória para geração determinística.
        """
        self.width = width
        self.height = height
        self.total_frames = total_frames
        self.seed = seed
        self._precompute_pitch()

    def _precompute_pitch(self) -> None:
        """Gera o gramado base com faixas de corte alternadas e marcações de campo."""
        pitch = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 1. Faixas de corte do gramado (duas tonalidades de verde futebol)
        band_width = 70
        c1 = np.array([36, 126, 44], dtype=np.uint8)  # Verde oliva rico (BGR)
        c2 = np.array([44, 142, 54], dtype=np.uint8)  # Verde grama vibrante (BGR)

        for col in range(0, self.width, band_width):
            color = c1 if (col // band_width) % 2 == 0 else c2
            pitch[:, col:min(self.width, col + band_width)] = color

        # 2. Textura sutil de grama e iluminação com gradiente radial (refletores de estádio)
        np.random.seed(self.seed)
        noise = np.random.randint(-6, 7, (self.height, self.width, 3), dtype=np.int16)
        pitch = np.clip(pitch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Gradiente radial de iluminação centralizada
        y_indices, x_indices = np.indices((self.height, self.width))
        cx_field, cy_field = self.width * 0.5, self.height * 0.45
        max_dist = math.sqrt((self.width * 0.5) ** 2 + (self.height * 0.5) ** 2)
        dist_map = np.sqrt((x_indices - cx_field) ** 2 + (y_indices - cy_field) ** 2) / max_dist
        vignette = 1.0 - 0.22 * dist_map  # atenuação suave nas bordas
        vignette = np.stack([vignette] * 3, axis=-1)
        pitch = np.clip(pitch * vignette, 0, 255).astype(np.uint8)

        # 3. Linhas oficiais de marcação de campo (brancas)
        line_color = (235, 235, 235)
        lw = 2

        # Linha lateral e de fundo (margem de 40px)
        margin = 35
        cv2.rectangle(pitch, (margin, margin), (self.width - margin, self.height - margin), line_color, lw, cv2.LINE_AA)

        # Grande área da grande área (penalty box)
        box_w = int(self.width * 0.45)
        box_h = int(self.height * 0.50)
        box_left = (self.width - box_w) // 2
        box_bottom = self.height - margin
        box_top = box_bottom - box_h
        cv2.rectangle(pitch, (box_left, box_top), (box_left + box_w, box_bottom), line_color, lw, cv2.LINE_AA)

        # Pequena área (área da meta / 6-yard box)
        small_w = int(box_w * 0.48)
        small_h = int(box_h * 0.35)
        small_left = (self.width - small_w) // 2
        small_top = box_bottom - small_h
        cv2.rectangle(pitch, (small_left, small_top), (small_left + small_w, box_bottom), line_color, lw, cv2.LINE_AA)

        # Ponto penal (marca de cal)
        penalty_spot_y = box_bottom - int(box_h * 0.60)
        cv2.circle(pitch, (self.width // 2, penalty_spot_y), 4, line_color, -1, cv2.LINE_AA)

        # Arco da grande área (semicírculo do pênalti)
        arc_radius = int(box_h * 0.32)
        cv2.ellipse(
            pitch,
            (self.width // 2, penalty_spot_y),
            (arc_radius, arc_radius),
            0,
            215,
            325,
            line_color,
            lw,
            cv2.LINE_AA,
        )

        self.base_pitch = pitch

    def _draw_player_silhouette(
        self,
        frame: np.ndarray,
        pos: Tuple[int, int],
        jersey_color: Tuple[int, int, int],
        is_moving_left: bool = True,
    ) -> None:
        """Desenha um jogador estilizado para contextualizar o campo e criar desafios de proximidade."""
        px, py = pos
        # Sombra no gramado
        cv2.ellipse(frame, (px, py + 22), (14, 5), 0, 0, 360, (20, 50, 25), -1, cv2.LINE_AA)
        # Tronco / camisa
        cv2.rectangle(frame, (px - 8, py - 10), (px + 8, py + 12), jersey_color, -1)
        # Shorts / calção
        cv2.rectangle(frame, (px - 8, py + 12), (px + 8, py + 18), (20, 20, 20), -1)
        # Pernas
        leg_offset = 3 if is_moving_left else -3
        cv2.line(frame, (px - 4, py + 18), (px - 4 - leg_offset, py + 24), (180, 150, 120), 2, cv2.LINE_AA)
        cv2.line(frame, (px + 4, py + 18), (px + 4 + leg_offset, py + 24), (180, 150, 120), 2, cv2.LINE_AA)
        # Cabeça
        cv2.circle(frame, (px, py - 16), 6, (180, 150, 120), -1, cv2.LINE_AA)

    def _draw_soccer_ball(
        self,
        frame: np.ndarray,
        cx: float,
        cy: float,
        radius: float = 12.0,
        height_above_ground: float = 0.0,
    ) -> None:
        """Desenha a bola de futebol com sombreamento esférico 3D, gomos pretos e sombra projetada."""
        icx, icy = int(round(cx)), int(round(cy))
        irad = max(4, int(round(radius)))

        # 1. Sombra projetada no gramado (offset proporcional à altura da bola)
        shadow_y = icy + int(round(height_above_ground * 0.85)) + 4
        shadow_rx = max(2, int(irad * (1.0 - height_above_ground * 0.008)))
        shadow_ry = max(1, int(shadow_rx * 0.38))
        cv2.ellipse(frame, (icx, shadow_y), (shadow_rx, shadow_ry), 0, 0, 360, (15, 45, 20), -1, cv2.LINE_AA)

        # 2. Corpo esférico da bola (base branca com sombreamento suave)
        ball_canvas = np.zeros((irad * 2 + 4, irad * 2 + 4, 3), dtype=np.uint8)
        bc_center = irad + 2

        # Gradiente esférico (simulando luz vindo do canto superior esquerdo)
        y_g, x_g = np.indices(ball_canvas.shape[:2])
        dist_center = np.sqrt((x_g - bc_center) ** 2 + (y_g - bc_center) ** 2)
        ball_mask = dist_center <= irad

        # Luz especular deslocada
        light_x, light_y = bc_center - irad * 0.35, bc_center - irad * 0.35
        light_dist = np.sqrt((x_g - light_x) ** 2 + (y_g - light_y) ** 2)
        shading = 255.0 - (light_dist / (irad * 1.8)) * 110.0
        shading = np.clip(shading, 140, 255).astype(np.uint8)

        ball_canvas[ball_mask] = np.stack([shading[ball_mask]] * 3, axis=-1)

        # 3. Gomos pretos característicos (pentágono central + gomos periféricos)
        patch_rad = max(2, int(irad * 0.32))
        cv2.circle(ball_canvas, (bc_center, bc_center), patch_rad, (25, 25, 25), -1, cv2.LINE_AA)

        # 4 gomos angulares ao redor
        for angle in [0, 72, 144, 216, 288]:
            rad_ang = math.radians(angle)
            gx = int(bc_center + math.cos(rad_ang) * (irad * 0.65))
            gy = int(bc_center + math.sin(rad_ang) * (irad * 0.65))
            cv2.circle(ball_canvas, (gx, gy), max(1, int(patch_rad * 0.65)), (30, 30, 30), -1, cv2.LINE_AA)

        # Contorno sutil
        cv2.circle(ball_canvas, (bc_center, bc_center), irad, (40, 40, 40), 1, cv2.LINE_AA)

        # 4. Sobreposição suave no quadro respeitando os limites
        top = max(0, icy - bc_center)
        bottom = min(frame.shape[0], icy - bc_center + ball_canvas.shape[0])
        left = max(0, icx - bc_center)
        right = min(frame.shape[1], icx - bc_center + ball_canvas.shape[1])

        c_top = top - (icy - bc_center)
        c_bottom = c_top + (bottom - top)
        c_left = left - (icx - bc_center)
        c_right = c_left + (right - left)

        crop_canvas = ball_canvas[c_top:c_bottom, c_left:c_right]
        crop_mask = ball_mask[c_top:c_bottom, c_left:c_right]

        frame_roi = frame[top:bottom, left:right]
        frame_roi[crop_mask] = crop_canvas[crop_mask]

    def get_frame(self, frame_idx: int) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Gera o quadro no instante especificado com a física exata da bola.

        Regra Física:
        Simula um passe ou finalização em curva (trajetória balística com sustentação aerodinâmica):
          X(t): movimento da esquerda para a direita com leve aceleração e curva em S.
          Y(t): arco parabólico cruzando a grande área.
          Z(t): elevação vertical da bola e retorno suave ao solo.

        Returns:
            Tupla (frame_bgr, bbox_real) onde bbox_real é (cx, cy, w, h).
        """
        frame = self.base_pitch.copy()
        t = frame_idx / max(1, self.total_frames - 1)

        # 1. Posições dos jogadores (dois defensores e um atacante na área)
        self._draw_player_silhouette(frame, (int(self.width * 0.35), int(self.height * 0.62)), (20, 20, 160))
        self._draw_player_silhouette(frame, (int(self.width * 0.58), int(self.height * 0.52)), (20, 20, 160))
        self._draw_player_silhouette(frame, (int(self.width * 0.72), int(self.height * 0.68)), (180, 40, 40), False)

        # 2. Posição da bola (física realista de cruzamento aéreo em curva)
        start_x, end_x = self.width * 0.18, self.width * 0.82
        start_y, end_y = self.height * 0.76, self.height * 0.38

        # Deslocamento horizontal suave
        cx = start_x + (end_x - start_x) * t + 28.0 * math.sin(t * math.pi)

        # Deslocamento vertical com efeito de arco parabólico
        apex_height = 95.0
        height_z = math.sin(t * math.pi) * apex_height
        cy = start_y + (end_y - start_y) * t - height_z

        # Variação sutil de escala com a proximidade/altura
        ball_radius = 11.0 + 3.0 * math.sin(t * math.pi)

        self._draw_soccer_ball(frame, cx, cy, radius=ball_radius, height_above_ground=height_z)

        # Bounding box real teórica
        bw = bh = ball_radius * 2.2
        real_bbox = (float(cx), float(cy), float(bw), float(bh))

        return frame, real_bbox


class VisualDemoComposer:
    """Monta composições visuais elegantes, ricas em telemetria, para demonstração acadêmica e técnica."""

    def __init__(
        self,
        output_width: int = 1280,
        output_height: int = 720,
    ):
        """Inicializa o compositor para resolução HD 1280x720."""
        self.out_w = output_width
        self.out_h = output_height

    def compose_dashboard_frame(
        self,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        preproc_frame: np.ndarray,
        tracker_bbox: Optional[Tuple[float, float, float, float]],
        last_roi: Optional[Dict[str, int]],
        frame_idx: int,
        total_frames: int,
        fps_metric: float = 61.8,
        ga_score: float = 0.6287,
        map50: float = 0.812,
        tracking_mode: str = "roi",
    ) -> np.ndarray:
        """Compõe um painel integrado com 3 vistas principais e HUD de telemetria.

        Layout:
          - Painel Esquerdo (820x620): Quadro do jogo anotado (bola, rastro da trajetória, ROI dinâmica).
          - Painel Superior Direito (420x295): Saída do pré-processamento adaptativo (Gamma 1.21 + HSV).
          - Painel Inferior Direito (420x295): Zoom focal da Região de Interesse (ROI 4.27x).
          - Barras Superior e Inferior: Título acadêmico, telemetria em tempo real e parâmetros do AG.
        """
        dashboard = np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
        dashboard[:] = (22, 24, 28)  # Fundo escuro moderno (Dark Theme)

        # 1. Barra de Cabeçalho Superior (altura 48px)
        header_h = 44
        cv2.rectangle(dashboard, (0, 0), (self.out_w, header_h), (14, 16, 20), -1)
        cv2.line(dashboard, (0, header_h), (self.out_w, header_h), (50, 56, 68), 1)

        # Título e badges do cabeçalho
        cv2.putText(
            dashboard,
            "SOCCER BALL DETECTOR | YOLO11 + GENETIC ALGORITHM (DEAP) DYNAMIC ROI TRACKER",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        badge_text = f"MODE: {tracking_mode.upper()}"
        badge_color = (0, 200, 100) if tracking_mode == "roi" else ((0, 165, 255) if tracking_mode == "full" else (80, 80, 220))
        cv2.rectangle(dashboard, (self.out_w - 200, 10), (self.out_w - 20, 34), badge_color, -1)
        cv2.putText(
            dashboard,
            badge_text,
            (self.out_w - 188, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )

        # 2. Painel Principal Esquerdo (Jogo com Rastreamento)
        main_x, main_y = 16, header_h + 12
        main_w = int(self.out_w * 0.63)
        main_h = self.out_h - header_h - 60
        resized_main = cv2.resize(annotated_frame, (main_w, main_h), interpolation=cv2.INTER_AREA)
        dashboard[main_y:main_y + main_h, main_x:main_x + main_w] = resized_main
        cv2.rectangle(dashboard, (main_x, main_y), (main_x + main_w, main_y + main_h), (50, 56, 68), 1)

        # Rótulo sobreposto no painel principal
        cv2.putText(
            dashboard,
            "1. Match Video Feed (YOLO11 Detection + Adaptive Trajectory History)",
            (main_x + 10, main_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 3. Painel Superior Direito: Pré-processamento adaptativo
        side_x = main_x + main_w + 16
        side_w = self.out_w - side_x - 16
        side_h = (main_h - 12) // 2

        preproc_y = main_y
        resized_preproc = cv2.resize(preproc_frame, (side_w, side_h), interpolation=cv2.INTER_AREA)
        dashboard[preproc_y:preproc_y + side_h, side_x:side_x + side_w] = resized_preproc
        cv2.rectangle(dashboard, (side_x, preproc_y), (side_x + side_w, preproc_y + side_h), (50, 56, 68), 1)

        cv2.putText(
            dashboard,
            "2. GA Preprocessing (Gamma=1.21, HSV Mask)",
            (side_x + 8, preproc_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 4. Painel Inferior Direito: Zoom na ROI Dinâmica
        roi_y = preproc_y + side_h + 12
        roi_panel = np.zeros((side_h, side_w, 3), dtype=np.uint8)
        roi_panel[:] = (30, 34, 40)

        if tracker_bbox is not None and last_roi is not None:
            r = last_roi
            crop = raw_frame[r["top"]:r["bottom"], r["left"]:r["right"]]
            if crop.size > 0:
                zoom_crop = cv2.resize(crop, (side_w - 4, side_h - 32), interpolation=cv2.INTER_LINEAR)
                # Bounding box central destacada no zoom
                zw, zh = zoom_crop.shape[1], zoom_crop.shape[0]
                bx1, by1 = int(zw * 0.35), int(zh * 0.35)
                bx2, by2 = int(zw * 0.65), int(zh * 0.65)
                cv2.rectangle(zoom_crop, (bx1, by1), (bx2, by2), (0, 255, 0), 2, cv2.LINE_AA)
                roi_panel[28:28 + zoom_crop.shape[0], 2:2 + zoom_crop.shape[1]] = zoom_crop
        else:
            cv2.putText(
                roi_panel,
                "ROI Offline - Full-Frame Fallback Active",
                (30, side_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (120, 130, 150),
                1,
                cv2.LINE_AA,
            )

        dashboard[roi_y:roi_y + side_h, side_x:side_x + side_w] = roi_panel
        cv2.rectangle(dashboard, (side_x, roi_y), (side_x + side_w, roi_y + side_h), (50, 56, 68), 1)

        cv2.putText(
            dashboard,
            "3. Dynamic ROI Window Focus (Scale 4.27x)",
            (side_x + 8, roi_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 5. Barra Inferior de Telemetria (altura 36px)
        footer_y = self.out_h - 36
        cv2.rectangle(dashboard, (0, footer_y), (self.out_w, self.out_h), (14, 16, 20), -1)
        cv2.line(dashboard, (0, footer_y), (self.out_w, footer_y), (50, 56, 68), 1)

        coord_str = f"({tracker_bbox[0]:.1f}, {tracker_bbox[1]:.1f})" if tracker_bbox else "N/A"
        telemetry_text = (
            f"Frame: {frame_idx:02d}/{total_frames:02d}  |  "
            f"Ball Centroid: {coord_str}  |  "
            f"Inference FPS: ~{fps_metric:.1f}  |  "
            f"Fitness Score: {ga_score:.4f}  |  "
            f"Final mAP50: {map50:.3f}  |  "
            f"Paper Ref: Table 3 (1533 ball detections)"
        )

        cv2.putText(
            dashboard,
            telemetry_text,
            (18, self.out_h - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (180, 190, 205),
            1,
            cv2.LINE_AA,
        )

        return dashboard

    def create_pipeline_comparison_image(
        self,
        raw_frame: np.ndarray,
        gamma_frame: np.ndarray,
        bg_removed_frame: np.ndarray,
        annotated_frame: np.ndarray,
    ) -> np.ndarray:
        """Gera uma figura estática 2x2 comparativa detalhando os quatro estágios do pipeline."""
        panel_w = 620
        panel_h = 340
        canvas_w = panel_w * 2 + 30
        canvas_h = panel_h * 2 + 80

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (20, 22, 26)

        # Cabeçalho
        cv2.putText(
            canvas,
            "SOCCER BALL DETECTOR - ADAPTIVE PREPROCESSING & TRACKING PIPELINE COMPARISON",
            (24, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        # 4 Estágios
        stages = [
            ("Stage 1: Raw Field Frame (Challenging Shadows & Contrast)", raw_frame, 10, 50),
            ("Stage 2: Adaptive Gamma Correction (gamma = 1.21)", gamma_frame, panel_w + 20, 50),
            ("Stage 3: HSV Color Segmentation (Background Filtering)", bg_removed_frame, 10, panel_h + 65),
            ("Stage 4: YOLO11 Detection + Dynamic ROI & Trajectory", annotated_frame, panel_w + 20, panel_h + 65),
        ]

        for title, img, px, py in stages:
            resized = cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            canvas[py:py + panel_h, px:px + panel_w] = resized
            cv2.rectangle(canvas, (px, py), (px + panel_w, py + panel_h), (60, 68, 80), 1)
            cv2.rectangle(canvas, (px, py), (px + panel_w, py + 26), (14, 16, 20), -1)
            cv2.putText(canvas, title, (px + 10, py + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 180), 1, cv2.LINE_AA)

        return canvas

    def save_gif(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 15,
        optimize_palette: bool = True,
        max_width: Optional[int] = 768,
        colors: int = 128,
    ) -> None:
        """Salva a sequência de quadros em formato GIF animado de alta fidelidade e tamanho otimizado.

        Regra de Engenharia:
        Redimensiona suavemente os quadros para largura amigável à web (padrão 768px),
        converte quadros BGR para RGB e aplica quantização de paleta adaptativa sem dither
        para preservar nitidez de texto, cores do gramado e manter o peso final abaixo de 3 MB.
        """
        pil_images = []
        duration_ms = int(1000 / fps)

        for f in frames:
            frame_to_proc = f
            if max_width is not None and f.shape[1] > max_width:
                scale = max_width / float(f.shape[1])
                new_h = int(f.shape[0] * scale)
                frame_to_proc = cv2.resize(f, (max_width, new_h), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame_to_proc, cv2.COLOR_BGR2RGB)
            p_img = Image.fromarray(rgb)
            if optimize_palette:
                p_img = p_img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
            pil_images.append(p_img)

        if pil_images:
            pil_images[0].save(
                output_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=duration_ms,
                loop=0,
                optimize=True,
            )
