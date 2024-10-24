from ultralytics import YOLO

def train_model():
    # Load a COCO-pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="yolo8/my_dataset_yolov5/data.yaml", epochs=2, imgsz=640, save=True)

if __name__ == "__main__":
    train_model()