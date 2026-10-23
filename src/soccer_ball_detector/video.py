import time
import cv2
from .detector import SoccerBallTracker


def annotate_video(input_path, output_path, model_path="yolov11-small.pt", config_path=None):
    """Processa um arquivo de video gravando anotacoes da bola detectada."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Nao foi possivel abrir video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = SoccerBallTracker(model_path=model_path, config_path=config_path)

    frame_count = 0
    detections = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        bbox = tracker.process_frame(frame)

        if bbox is not None:
            detections += 1
            cx, cy, bw, bh = bbox
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    elapsed = time.time() - start

    return {
        "frames": frame_count,
        "detections": detections,
        "time_seconds": elapsed,
        "effective_fps": frame_count / elapsed if elapsed > 0 else 0
    }
