from ultralytics import YOLO 
model=YOLO("yolo11n.pt")

model.train(data="data.yaml",epochs=200,imgsz=640,batch=64)