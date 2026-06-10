from ultralytics import YOLO

YOLO("yolo26x-pose.pt").export(format="engine", half=True, imgsz=640)