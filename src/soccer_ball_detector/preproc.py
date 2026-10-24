"""Módulo de pré-processamento de imagens para detecção de bola em partidas de futebol.

Motivação Acadêmica e de Engenharia:
A bola de futebol é um objeto de pequenas dimensões relativas (frequentemente < 0.5% da área do quadro),
sujeito a desfoque de movimento, sombras de arquibancada e refletores de estádio.
Este módulo reúne transformações fotométricas e segmentações em espaços de cor (HSV, LAB, YUV)
cujas combinações e hiperparâmetros foram explorados evolutivamente via Algoritmo Genético (DEAP)
para maximizar o mAP50 da rede YOLO11 sem comprometer a taxa de quadros (FPS).
"""

import cv2
import numpy as np


def remove_bg(image, lower_bound=(100, 100, 100), upper_bound=(140, 255, 255)):
    """Remove pixels dentro da faixa HSV especificada, zerando elementos do fundo.

    Regra de Negócio:
    Em jogos de futebol, a segmentação por matiz (Hue) permite mascarar a tonalidade dominante
    do gramado ou uniformes estáticos, realçando o contraste da bola branca/colorida.

    Args:
        image: Imagem BGR (numpy.ndarray de uint8).
        lower_bound: Limite inferior HSV (H: 0-180, S: 0-255, V: 0-255).
        upper_bound: Limite superior HSV (H: 0-180, S: 0-255, V: 0-255).

    Returns:
        Imagem BGR com pixels da faixa mascarados em preto [0, 0, 0].
    """
    if image is None or image.size == 0:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_bound, dtype="uint8"), np.array(upper_bound, dtype="uint8"))
    out = image.copy()
    out[mask != 0] = [0, 0, 0]
    return out


def hist_eq(image):
    """Equaliza o canal de luminância Y no espaço de cor YUV.

    Regra de Negócio:
    A equalização direta nos três canais BGR distorce o equilíbrio cromático. A conversão para
    YUV e equalização exclusiva do canal Y acentua contrastes globais preservando as cores originais.

    Args:
        image: Imagem BGR (numpy.ndarray).

    Returns:
        Imagem BGR equalizada.
    """
    if image is None or image.size == 0:
        return image
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) no canal L do espaço LAB.

    Regra de Negócio:
    O CLAHE divide a imagem em blocos (tiles) e equaliza localmente com limitação de inclinação
    (clipLimit), prevenindo a amplificação excessiva de ruído em áreas uniformes como o gramado.

    Args:
        image: Imagem BGR (numpy.ndarray).
        clip_limit: Limite de corte de contraste (padrão: 2.0).
        tile_grid_size: Tupla (linhas, colunas) do grid adaptativo (ex: (8, 8) ou (16, 16)).

    Returns:
        Imagem BGR com contraste adaptativo local aprimorado.
    """
    if image is None or image.size == 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe_obj.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gamma_correction(image, gamma=1.0):
    """Aplica correção gamma não linear por Look-Up Table (LUT).

    Regra de Negócio:
    Equação: I_out = 255 * (I_in / 255) ^ (1 / gamma).
    Para gamma > 1, atenua sombras e realça detalhes escuros; para gamma < 1, atenua superexposição.
    A pré-computação de uma LUT de 256 posições garante execução em tempo real em alta resolução.

    Args:
        image: Imagem BGR (numpy.ndarray).
        gamma: Expoente gamma positivo (configuração ótima do AG: gamma ≈ 1.211).

    Returns:
        Imagem BGR corrigida via LUT.
    """
    if image is None or image.size == 0:
        return image
    if gamma <= 0:
        gamma = 1.0
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_pipeline(image, use_hist_eq=False, use_clahe=False, clahe_clip=2.0,
                   tile_size=(8, 8), use_gamma=False, gamma_value=1.0,
                   use_remove_bg=False, lower_bg=(100, 100, 100), upper_bg=(140, 255, 255)):
    """Executa a composição sequencial das etapas de pré-processamento.

    Regra de Negócio:
    Permite encadear transformações fotométricas de forma idempotente e modular,
    espelhando a pipeline avaliada durante o processo evolutivo do Algoritmo Genético.

    Args:
        image: Imagem de entrada BGR.
        use_hist_eq: Habilita equalização de histograma YUV.
        use_clahe: Habilita CLAHE em LAB.
        clahe_clip: Limiar de clip para CLAHE.
        tile_size: Grid size do CLAHE.
        use_gamma: Habilita correção gamma.
        gamma_value: Valor do parâmetro gamma.
        use_remove_bg: Habilita máscara HSV de fundo.
        lower_bg: Limite inferior da cor de fundo em HSV.
        upper_bg: Limite superior da cor de fundo em HSV.

    Returns:
        Imagem BGR processada.
    """
    if image is None or image.size == 0:
        return image

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
