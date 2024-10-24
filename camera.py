from ultralytics import YOLO
import cv2
import math




cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Загружаем модель
model = YOLO(r"runs\detect\train7\weights\best.pt")

# Классы объектов
classNames = ["Phone", "human face"
              ]

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    # Список объектов
    objects = []

    # Обрабатываем результаты
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Уровень доверия
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Название класса
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Добавляем объект в список
            objects.append({
                "class": class_name,
                "confidence": confidence,
                "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, f"{class_name} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255),
                        2)

    detected_objects = objects

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



