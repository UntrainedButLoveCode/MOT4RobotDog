from ultralytics import YOLO

model = YOLO("yolo11l.pt")

model.export(format="engine",dynamic=True,half=True)