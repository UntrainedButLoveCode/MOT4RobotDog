from ultralytics import YOLO
model = YOLO("yolo11n.pt")
res = model.predict(source=r"C:\Users\kakar\Desktop\杰创\test.jpeg",save=True)