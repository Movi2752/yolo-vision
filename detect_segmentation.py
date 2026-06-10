"""
Фаза 1 — сегментация объектов с веб-камеры на YOLO26 (TensorRT engine).
Вместо рамок — точные маски объектов.
Потоковая архитектура сохранена.

Запуск:  python detect_segmentation.py
Выход:   клавиша q
"""
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

# --- настройки ---
MODEL = "yolo26x-seg.pt"   # сегментационная модель (или .pt)
CAM_INDEX = 0
CONF = 0.35
DEVICE = 0
MIRROR = True
WIN = "YOLO26 - segmentation (q to quit)"


def letterbox_to_window(img, win_name):
    # (функция без изменений, как в исходнике)
    _, _, w, h = cv2.getWindowImageRect(win_name)
    if w <= 0 or h <= 0:
        return img
    ih, iw = img.shape[:2]
    if (w, h) == (iw, ih):
        return img
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


class Mailbox:
    # (код идентичен исходному)
    def __init__(self):
        self._cond = threading.Condition()
        self._item = None
        self._seq = 0

    def put(self, item):
        with self._cond:
            self._item = item
            self._seq += 1
            self._cond.notify_all()

    def get(self, last_seq, timeout=1.0):
        with self._cond:
            self._cond.wait_for(lambda: self._seq > last_seq, timeout=timeout)
            if self._seq > last_seq:
                return self._item, self._seq
            return None, last_seq


class CaptureThread(threading.Thread):
    def __init__(self, cap, out_box: Mailbox):
        super().__init__(daemon=True)
        self.cap = cap
        self.out = out_box
        self.alive = True

    def run(self):
        while self.alive:
            ok, frame = self.cap.read()
            if not ok:
                self.alive = False
                break
            self.out.put(frame)

    def stop(self):
        self.alive = False


class InferSegmentationThread(threading.Thread):
    def __init__(self, model, in_box: Mailbox, out_box: Mailbox):
        super().__init__(daemon=True)
        self.model = model
        self.inp = in_box
        self.out = out_box
        self.alive = True
        self.fps = 0.0

    def run(self):
        seq = 0
        prev = time.time()
        while self.alive:
            frame, seq = self.inp.get(seq)
            if frame is None:
                continue
            if MIRROR:
                frame = cv2.flip(frame, 1)

            # Инференс сегментационной модели
            results = self.model.predict(frame, conf=CONF, device=DEVICE, verbose=False)
            # plot() автоматически рисует маски (полупрозрачные заливки + контуры)
            annotated = results[0].plot()

            now = time.time()
            inst = 1.0 / (now - prev) if now > prev else 0.0
            prev = now
            self.fps = 0.9 * self.fps + 0.1 * inst
            cv2.putText(annotated, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            self.out.put(annotated)

    def stop(self):
        self.alive = False


def main():
    model = YOLO(MODEL, task="segment")   # task="segment"
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру #{CAM_INDEX}")

    raw_box, out_box = Mailbox(), Mailbox()
    capture = CaptureThread(cap, raw_box)
    infer = InferSegmentationThread(model, raw_box, out_box)
    capture.start()
    infer.start()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    seq = 0
    try:
        while capture.alive:
            annotated, seq = out_box.get(seq)
            if annotated is not None:
                cv2.imshow(WIN, letterbox_to_window(annotated, WIN))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        infer.stop()
        capture.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()