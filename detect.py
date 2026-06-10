"""
Фаза 1 — детектор объектов с веб-камеры на YOLO26 (TensorRT engine).
Конвейерная архитектура: захват / инференс / отображение в трёх потоках.

Запуск:  python detect.py
Выход:   клавиша q (в окне с видео)

Почему потоки помогают, несмотря на GIL: cap.read(), cv2.* и инференс
TensorRT — нативный код, который отпускает GIL, так что три стадии
реально работают параллельно на разных ядрах.

Архитектура:
  [CaptureThread]  камера -> mailbox raw      (всегда хранит ТОЛЬКО свежий кадр)
  [InferThread]    raw -> flip -> predict -> plot -> mailbox annotated
  [main]           annotated -> imshow + клавиши (UI обязан жить в main-потоке)

Mailbox на 1 слот вместо очереди: старые кадры перезаписываются,
поэтому нет накопления лага — детекция всегда про "сейчас".
"""
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

# --- настройки (крути под себя) ---
MODEL = "yolo26x.engine"   # engine собирается скриптом export_trt.py под конкретную GPU
CAM_INDEX = 0
CONF = 0.35                # 0.5 жестковат — режет валидные объекты
DEVICE = 0
MIRROR = False              # зеркало (выключи, если приложение камеры зеркалит само)
# Разрешение/FPS не задаём через cap.set(): Iriun — виртуальная камера,
# она игнорирует CAP_PROP_* и отдаёт то, что выставлено в приложении на телефоне.
WIN = "YOLO26 - detection (q to quit)"


def letterbox_to_window(img, win_name):
    """Вписывает кадр в окно с чёрными полями, сохраняя пропорции при ресайзе окна."""
    _, _, w, h = cv2.getWindowImageRect(win_name)
    if w <= 0 or h <= 0:
        return img
    ih, iw = img.shape[:2]
    if (w, h) == (iw, ih):          # окно совпадает с кадром — ресайз не нужен
        return img
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


class Mailbox:
    """Один слот под последнее значение. put() перезаписывает, get() ждёт новое."""

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
        """Блокируется, пока не появится элемент новее last_seq.
        Возвращает (item, seq) или (None, last_seq) по таймауту."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > last_seq, timeout=timeout)
            if self._seq > last_seq:
                return self._item, self._seq
            return None, last_seq


class CaptureThread(threading.Thread):
    """Молотит cap.read() в своём темпе, складывает свежий кадр в mailbox."""

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


class InferThread(threading.Thread):
    """Берёт свежий кадр, гоняет через модель, кладёт размеченный кадр + FPS."""

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
            if frame is None:        # таймаут — камера молчит, проверяем alive
                continue
            if MIRROR:
                frame = cv2.flip(frame, 1)

            results = self.model.predict(frame, conf=CONF, device=DEVICE, verbose=False)
            annotated = results[0].plot()

            now = time.time()
            inst = 1.0 / (now - prev) if now > prev else 0.0
            prev = now
            self.fps = 0.9 * self.fps + 0.1 * inst  # сглаживание, чтобы цифра не дёргалась

            cv2.putText(annotated, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            self.out.put(annotated)

    def stop(self):
        self.alive = False


def main() -> None:
    model = YOLO(MODEL, task="detect")

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру #{CAM_INDEX}")

    raw_box, out_box = Mailbox(), Mailbox()
    capture = CaptureThread(cap, raw_box)
    infer = InferThread(model, raw_box, out_box)
    capture.start()
    infer.start()

    # адаптивное окно: можно растягивать мышкой, пропорции сохраняет letterbox
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    seq = 0
    try:
        while capture.alive:
            annotated, seq = out_box.get(seq)
            if annotated is not None:
                cv2.imshow(WIN, letterbox_to_window(annotated, WIN))
            # сюда позже легко вшить режим "опиши кадр" через Gemini:
            # if cv2.waitKey(1) & 0xFF == ord(" "):  describe_with_vlm(...)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        infer.stop()
        capture.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()