"""
Фаза 1 — базовый детектор объектов с веб-камеры на YOLO26.

Запуск:  python detect.py
Выход:   клавиша q (в окне с видео)

Модель знает ~80 классов COCO из коробки (люди, машины, гаджеты, посуда,
животные и т.д.) — это наша база. Дообучение под свою тему будет в Фазе 2.
"""
import time

import cv2
from ultralytics import YOLO
import numpy as np

# --- настройки (крути под себя) ---
MODEL = "yolo26n.pt"   # nano — самый лёгкий вариант; скачается сам при первом запуске
CAM_INDEX = 0          # 0 — первая/встроенная камера; поставь 1, 2... если камер несколько
CONF = 0.5             # порог уверенности (0..1): ниже — больше детекций, но больше мусора
DEVICE = 0             # 0 — твоя NVIDIA GPU; "cpu" — принудительно на процессоре

def letterbox_to_window(img, win_name):
    _, _, w, h = cv2.getWindowImageRect(win_name)
    if w <= 0 or h <= 0:
        return img
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas

def main() -> None:
    model = YOLO(MODEL)

    # CAP_DSHOW заметно быстрее открывает камеру на Windows
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру #{CAM_INDEX}")

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # зеркало по горизонтали

        # инференс по одному кадру
        results = model.predict(frame, conf=CONF, device=DEVICE, verbose=False)
        annotated = results[0].plot()  # кадр с нарисованными рамками и подписями (BGR)

        # FPS в углу
        now = time.time()
        fps = 1.0 / (now - prev) if now > prev else 0.0
        prev = now
        cv2.putText(
            annotated, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
        )

        WIN = "YOLO26 - detection (q to quit)"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(WIN, letterbox_to_window(annotated, WIN))

        # сюда позже легко вшить режим "опиши кадр" через Gemini:
        # if cv2.waitKey(1) & 0xFF == ord(" "):  describe_with_vlm(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()