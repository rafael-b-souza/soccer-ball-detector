import numpy as np


def extract_ball_box(results, img, threshold=0.3):
    """Extrai coordenadas (cx, cy, w, h) da primeira deteccao acima do limiar."""
    result = results[0] if isinstance(results, (list, tuple)) else results
    if not hasattr(result, "boxes") or result.boxes is None or len(result.boxes) == 0:
        return None

    for box in result.boxes:
        conf = box.conf.item() if hasattr(box, "conf") and box.conf is not None else 1.0
        if conf < threshold:
            continue
        xywh = box.xywh[0]
        if hasattr(xywh, "cpu"):
            xywh = xywh.cpu().numpy()
        else:
            xywh = np.array(xywh)
        return float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3])
    return None


def get_roi_from_bbox(bbox, image_shape, roi_scale=3.0):
    """Calcula retangulo delimitador da regiao de interesse (ROI)."""
    height, width = image_shape[:2]
    cx, cy, w, h = bbox
    roi_w = w * roi_scale
    roi_h = h * roi_scale

    left = int(max(0, cx - roi_w / 2))
    top = int(max(0, cy - roi_h / 2))
    right = int(min(width, cx + roi_w / 2))
    bottom = int(min(height, cy + roi_h / 2))

    return {"left": left, "top": top, "right": right, "bottom": bottom}
