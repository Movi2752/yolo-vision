from ultralytics import YOLO

YOLO("yolo26m.pt").export(format="engine", half=True, imgsz=960, fraction=1.0)