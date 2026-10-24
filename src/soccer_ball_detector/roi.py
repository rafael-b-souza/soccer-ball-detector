"""Módulo de extração e manipulação de Regiões de Interesse (ROI) e caixas delimitadoras.

Motivação Acadêmica e de Engenharia:
A detecção contínua da bola de futebol no quadro completo de alta resolução (ex: 1280x720 ou 1920x1080)
impõe um custo computacional desnecessário e amplia a taxa de falsos positivos provocados por elementos
de fundo (jogadores, chuteiras brancas, placas de publicidade e faixas do gramado).
O cálculo da ROI dinâmica delimita a busca ao redor da última posição confirmada com um fator de escala
otimizado por Algoritmo Genético (roi_scale ≈ 4.27), reduzindo a latência por quadro. Caso o rastreador
perca o objeto dentro da ROI, o sistema aciona fallback automático para o quadro total.
"""

import numpy as np


def extract_ball_box(results, img=None, threshold=0.3, target_classes=None):
    """Extrai coordenadas (cx, cy, w, h) da detecção mais confiável da bola acima do limiar.

    Regra de Negócio:
    Prioriza a detecção de maior confiança da classe correspondente à bola (classe 'bola' em modelo
    customizado ou 'sports ball' / classe 32 em pesos pré-treinados COCO). Caso target_classes seja
    especificado, filtra estritamente por esses IDs de classe.

    Args:
        results: Objeto de predição do Ultralytics YOLO ou lista de resultados.
        img: Imagem de referência (opcional, mantida para compatibilidade).
        threshold: Limiar mínimo de confiança para aceitação da bounding box.
        target_classes: Coleção opcional de inteiros com os IDs de classe permitidos.

    Returns:
        Tupla (cx, cy, w, h) com as coordenadas centrais e dimensões da bola, ou None se não encontrada.
    """
    result = results[0] if isinstance(results, (list, tuple)) else results
    if not hasattr(result, "boxes") or result.boxes is None or len(result.boxes) == 0:
        return None

    # Se target_classes não for fornecido, tenta inferir classe pelo mapa de nomes se disponível
    allowed_classes = None
    if target_classes is not None:
        allowed_classes = set(target_classes)
    elif hasattr(result, "names") and isinstance(result.names, dict):
        # Se contiver 'sports ball' ou 'bola', prioriza essas classes
        inferred = [cid for cid, cname in result.names.items() if cname.lower() in ("sports ball", "bola", "ball")]
        if inferred:
            allowed_classes = set(inferred)

    best_box = None
    max_conf = -1.0

    for box in result.boxes:
        conf = box.conf.item() if hasattr(box, "conf") and box.conf is not None else 1.0
        if conf < threshold:
            continue

        if allowed_classes is not None and hasattr(box, "cls") and box.cls is not None:
            cls_id = int(box.cls.item())
            if cls_id not in allowed_classes:
                continue

        if conf > max_conf:
            max_conf = conf
            xywh = box.xywh[0]
            if hasattr(xywh, "cpu"):
                xywh = xywh.cpu().numpy()
            else:
                xywh = np.array(xywh)
            best_box = (float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3]))

    return best_box


def get_roi_from_bbox(bbox, image_shape, roi_scale=3.0):
    """Calcula o retângulo delimitador da Região de Interesse (ROI) dinâmica.

    Regra de Negócio:
    Expande a bounding box da bola pela escala roi_scale selecionada pelo Algoritmo Genético,
    garantindo que o deslocamento inter-frames da bola permaneça dentro da janela de busca.
    Aplica truncamento nas bordas da imagem para prevenir coordenadas negativas ou além dos limites.

    Args:
        bbox: Tupla (cx, cy, w, h) da bounding box no espaço de coordenadas atual.
        image_shape: Shape da imagem (altura, largura, ...).
        roi_scale: Fator multiplicador das dimensões da bounding box (padrão ótimo do AG: ~4.27).

    Returns:
        Dicionário com limites absolutos: {'left': int, 'top': int, 'right': int, 'bottom': int}.
    """
    height, width = image_shape[:2]
    cx, cy, w, h = bbox
    roi_w = w * roi_scale
    roi_h = h * roi_scale

    left = int(max(0, cx - roi_w / 2))
    top = int(max(0, cy - roi_h / 2))
    right = int(min(width, cx + roi_w / 2))
    bottom = int(min(height, cy + roi_h / 2))

    return {"left": left, "top": top, "right": right, "bottom": bottom}
