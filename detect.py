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
import torch
import threading

# --- настройки (крути под себя) ---
MODEL = "yolo26m.engine"
CAM_INDEX = 0
CONF = 0.35            # 0.5 жестковат — режет валидные объекты; мусор почти не вырастет
DEVICE = 0

class CamReader:
    def __init__(self, cap):
        self.cap, self.frame, self.ok = cap, None, True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()
        # ждём первый кадр, чтобы main не стартовал с пустотой
        t0 = time.time()
        while self.frame is None and self.ok and time.time() - t0 < 5:
            time.sleep(0.01)

    def _loop(self):
        while self.ok:
            ok, f = self.cap.read()
            if not ok:
                self.ok = False
                break
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return (self.frame is not None), (self.frame.copy() if self.frame is not None else None)

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
    print(
        f"🚀 GPU Active: {torch.cuda.is_available()} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    model = YOLO(MODEL)

    # CAP_DSHOW заметно быстрее открывает камеру на Windows
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру #{CAM_INDEX}")
    reader = CamReader(cap)

    prev = time.time()
    while True:
        ok, frame = reader.read()
        if not reader.ok:
            break  # камера реально умерла
        if not ok:
            continue  # кадра пока нет — пропускаем итерацию
        frame = cv2.flip(frame, 1)  # зеркало по горизонталиq

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