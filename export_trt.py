from ultralytics import YOLO

YOLO("yolo26s.pt").export(format="engine", half=True, imgsz=640)