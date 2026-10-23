import json
import math
import time
import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from .preproc import apply_pipeline
from .roi import extract_ball_box, get_roi_from_bbox


class SoccerBallTracker:
    """Rastreador de bola com YOLO e estagio duplo (ROI adaptativa + fallback)."""

    def __init__(self, model_path="yolov11-small.pt", config_path=None, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(model_path).to(self.device)
        self.params = {
            "preproc": "gamma",
            "remove_bg": True,
            "lower_blue": 106,
            "upper_blue": 156,
            "clahe_clip": 2.8,
            "tile_size": (16, 16),
            "gamma_value": 1.21,
            "roi_scale": 4.27
        }

        if config_path:
            with open(config_path, "r") as f:
                loaded = json.load(f)
                self.params.update(loaded)

        self.last_bbox = None

    def _prepare_tensor(self, img_bgr):
        tensor = torch.tensor(img_bgr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        _, _, h, w = tensor.shape
        pad_h = (math.ceil(h / 32) * 32) - h
        pad_w = (math.ceil(w / 32) * 32) - w
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)
        return tensor

    def process_frame(self, frame):
        """Processa um frame retornando a bounding box absoluta (cx, cy, w, h) ou None."""
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
        if self.last_bbox is not None:
            roi_coords = get_roi_from_bbox(self.last_bbox, frame.shape, roi_scale)
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
                    upper_bg=upper_bg
                )
                crop_tensor = self._prepare_tensor(preproc_crop)
                results = self.model(crop_tensor, verbose=False)
                box = extract_ball_box(results, preproc_crop)
                if box is not None:
                    cx, cy, bw, bh = box
                    self.last_bbox = (cx + roi_coords["left"], cy + roi_coords["top"], bw, bh)
                    return self.last_bbox

        # Fallback: executa inferencia no frame completo
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
            upper_bg=upper_bg
        )
        full_tensor = self._prepare_tensor(preproc_full)
        results = self.model(full_tensor, verbose=False)
        box = extract_ball_box(results, preproc_full)
        self.last_bbox = box
        return self.last_bbox
