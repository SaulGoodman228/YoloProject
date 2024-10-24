from ultralytics import YOLO

model = YOLO('yolov8m.pt')

result = model.train(data='my_dataset_yolov5/data.yaml',epochs=100, imgsz=640)