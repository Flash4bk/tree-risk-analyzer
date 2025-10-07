from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import os
import shutil
import requests
import torch

app = FastAPI(title="Tree Risk Analyzer — AI Forest Tool")

# ======= Пути к моделям =======
TREE_MODEL_PATH = "tree_model.pt"
STICK_MODEL_PATH = "stick_model.pt"
CLASSIFIER_PATH = "classifier.pth"

# ======= Пути данных =======
DATASET_PATH = "datasets/new_train"
IMAGE_UPLOAD_PATH = os.path.join(DATASET_PATH, "images")
MASK_UPLOAD_PATH = os.path.join(DATASET_PATH, "masks")
os.makedirs(IMAGE_UPLOAD_PATH, exist_ok=True)
os.makedirs(MASK_UPLOAD_PATH, exist_ok=True)

# ======= OpenWeather =======
OPENWEATHER_API_KEY = "dc825ffd002731568ec7766eafb54bc9"  # ← вставь сюда API ключ из openweathermap.org
# ======= Yandex Disk =======
YANDEX_TOKEN = "ТВОЙ_OAUTH_ТОКЕН"  # вставь, если хочешь выгрузку

# ======= МОДЕЛИ =======
print("Загрузка YOLO моделей...")
yolo_tree = YOLO(TREE_MODEL_PATH)
yolo_stick = YOLO(STICK_MODEL_PATH)
print("Модели загружены!")


# =========================================================
# =============== ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ =================
# =========================================================

def get_weather(lat: float, lon: float):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        r = requests.get(url)
        data = r.json()
        wind_speed = data["wind"]["speed"]
        wind_deg = data["wind"]["deg"]
        return {"wind_speed": wind_speed, "wind_deg": wind_deg}
    except Exception as e:
        return {"error": str(e)}


def calculate_risk(height_m, crown_width_m, trunk_diameter_m, wind_speed):
    try:
        ratio = crown_width_m / max(trunk_diameter_m, 0.01)
        wind_factor = wind_speed / 10
        risk = (ratio * wind_factor * (height_m / 10))
        risk = min(risk, 1.0)
        if risk < 0.3:
            category = "Low"
        elif risk < 0.7:
            category = "Medium"
        else:
            category = "High"
        return {"risk": round(risk, 2), "category": category}
    except Exception as e:
        return {"error": str(e)}


def upload_to_yandex(filename: str, file_path: str):
    """Загрузка файла на Яндекс Диск"""
    try:
        headers = {"Authorization": f"OAuth {YANDEX_TOKEN}"}
        url = "https://cloud-api.yandex.net/v1/disk/resources/upload"
        params = {"path": f"/TreeApp/{filename}", "overwrite": "true"}
        upload_url = requests.get(url, headers=headers, params=params).json()["href"]
        with open(file_path, "rb") as f:
            requests.put(upload_url, files={"file": f})
        return True
    except Exception as e:
        print("Yandex upload error:", e)
        return False


# =========================================================
# ======================= /analyze ========================
# =========================================================

@app.post("/analyze")
async def analyze_tree(
    file: UploadFile = File(...),
    lat: float = Form(0.0),
    lon: float = Form(0.0),
    stick_length_m: float = Form(1.0),
    manual_stick: str = Form(None)
):
    """Основной анализ дерева"""
    try:
        # Сохраняем фото
        image_data = await file.read()
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        local_path = os.path.join(IMAGE_UPLOAD_PATH, file.filename)
        cv2.imwrite(local_path, img)

        # --- ПАЛКА ---
        results_stick = yolo_stick(img)
        stick_boxes = results_stick[0].boxes.xyxy.cpu().numpy()
        if len(stick_boxes) > 0:
            x1, y1, x2, y2 = stick_boxes[0]
            stick_length_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        elif manual_stick:
            pts = manual_stick.split(",")
            x1, y1, x2, y2 = map(float, pts)
            stick_length_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        else:
            stick_length_px = None

        if stick_length_px:
            scale_m_per_px = stick_length_m / stick_length_px
        else:
            scale_m_per_px = 0.002  # запасное значение

        # --- ДЕРЕВО ---
        results_tree = yolo_tree(img)
        boxes = results_tree[0].boxes.xyxy.cpu().numpy()
        if len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0]
            height_px = y2 - y1
            crown_width_px = x2 - x1
            height_m = height_px * scale_m_per_px
            crown_width_m = crown_width_px * scale_m_per_px
            trunk_diameter_m = 0.1 * height_m
        else:
            height_m = crown_width_m = trunk_diameter_m = 0

        # --- Погода и риск ---
        weather = get_weather(lat, lon)
        wind_speed = weather.get("wind_speed", 5.0)
        risk = calculate_risk(height_m, crown_width_m, trunk_diameter_m, wind_speed)

        # --- Загрузка в облако ---
        if YANDEX_TOKEN != "ТВОЙ_OAUTH_ТОКЕН":
            upload_to_yandex(file.filename, local_path)

        return {
            "scale_m_per_px": round(scale_m_per_px, 6),
            "height_m": round(height_m, 2),
            "crown_width_m": round(crown_width_m, 2),
            "trunk_diameter_m": round(trunk_diameter_m, 2),
            "weather": weather,
            "risk": risk
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# =========================================================
# ================= /upload_training_sample ===============
# =========================================================

@app.post("/upload_training_sample")
async def upload_training_sample(file: UploadFile = File(...)):
    """Загружает фото для обучения"""
    try:
        file_path = os.path.join(IMAGE_UPLOAD_PATH, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"status": "uploaded", "filename": file.filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# =========================================================
# ======================= /save_polygon ===================
# =========================================================

@app.post("/save_polygon")
async def save_polygon(data: dict = Body(...)):
    """Сохраняет контур (маску), который нарисовал пользователь"""
    try:
        filename = data["filename"]
        points = np.array(data["points"], np.int32)
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        save_path = os.path.join(MASK_UPLOAD_PATH, f"{filename.split('.')[0]}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, mask)
        return {"status": "mask_saved", "path": save_path}
    except Exception as e:
        return {"error": str(e)}


# =========================================================
# ======================== /train_now =====================
# =========================================================

@app.post("/train_now")
async def train_now():
    """Запускает дообучение YOLO"""
    try:
        model = YOLO(TREE_MODEL_PATH)
        results = model.train(
            data=f"{DATASET_PATH}/data.yaml",
            epochs=3,
            imgsz=640,
            device="cpu"
        )
        return {"status": "success", "details": str(results)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
