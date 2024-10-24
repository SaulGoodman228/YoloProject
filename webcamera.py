from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import cv2
import math
import threading
from starlette.requests import Request

app = FastAPI()

# Настройка Jinja2 для работы с шаблонами
templates = Jinja2Templates(directory="templates")

# Объекты, которые будут отображаться на странице
detected_objects = []
cap = cv2.VideoCapture(0)

# Обработка объектов с камеры
def detect_objects():
    global detected_objects
    model = YOLO(r"runs\detect\train7\weights\best.pt")

    classNames = ["Phone", "human face"]

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        objects = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]

                objects.append({
                    "class": class_name,
                    "confidence": confidence,
                    "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img, f"{class_name} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        detected_objects = objects
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Функция для передачи видеопотока
def video_stream():
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Запуск потока для распознавания объектов
threading.Thread(target=detect_objects, daemon=True).start()

# Маршрут для видеопотока
@app.get("/video")
def get_video():
    return StreamingResponse(video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

# Главная страница с объектами
@app.get("/", response_class=JSONResponse)
async def get_objects():
    return {"objects": detected_objects}

# Асинхронная функция для загрузки HTML-шаблона
@app.get("/webpage")
async def video_feed(request: Request):
    return templates.TemplateResponse("data.html", {"request": request})
