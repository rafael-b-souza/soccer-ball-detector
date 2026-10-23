import cv2
import numpy as np


def remove_bg(image, lower_bound=(100, 100, 100), upper_bound=(140, 255, 255)):
    """Remove pixels dentro da faixa HSV especificada."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_bound, dtype="uint8"), np.array(upper_bound, dtype="uint8"))
    out = image.copy()
    out[mask != 0] = [0, 0, 0]
    return out


def hist_eq(image):
    """Equaliza o canal Y no espaço YUV."""
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Aplica CLAHE adaptativo no canal L do espaco LAB."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe_obj.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gamma_correction(image, gamma=1.0):
    """Aplica correcao gamma via Look-Up Table."""
    if gamma <= 0:
        gamma = 1.0
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_pipeline(image, use_hist_eq=False, use_clahe=False, clahe_clip=2.0,
                   tile_size=(8, 8), use_gamma=False, gamma_value=1.0,
                   use_remove_bg=False, lower_bg=(100, 100, 100), upper_bg=(140, 255, 255)):
    """Executa a composicao de etapas de pre-processamento em uma imagem."""
    out = image.copy()

    if use_hist_eq:
        out = hist_eq(out)

    if use_clahe:
        out = clahe(out, clip_limit=clahe_clip, tile_grid_size=tile_size)

    if use_gamma and gamma_value != 1.0:
        out = gamma_correction(out, gamma=gamma_value)

    if use_remove_bg:
        out = remove_bg(out, lower_bound=lower_bg, upper_bound=upper_bg)

    return out
