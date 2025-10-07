from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import torch
import numpy as np
import cv2
import os
import shutil
import requests

app = FastAPI(title="Tree Risk Analyzer API")

# --- Пути к моделям ---
TREE_MODEL_PATH = "tree_model.pt"
STICK_MODEL_PATH = "stick_model.pt"
CLASSIFIER_PATH = "classifier.pth"

# --- Пути для обучения ---
DATASET_PATH = "datasets/new_train"
IMAGE_UPLOAD_PATH = os.path.join(DATASET_PATH, "images")
os.makedirs(IMAGE_UPLOAD_PATH, exist_ok=True)

# --- OpenWeather API ---
OPENWEATHER_API_KEY = "dc825ffd002731568ec7766eafb54bc9"  # ← вставь сюда API ключ из openweathermap.org


# ======== ФУНКЦИИ =========
def get_weather(lat: float, lon: float):
    """Получает погоду (ветер, направление) по координатам"""
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
    """Простая модель риска падения дерева"""
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
        return {"risk": risk, "category": category}
    except Exception as e:
        return {"error": str(e)}


# ======== ЗАГРУЗКА МОДЕЛЕЙ =========
print("Загрузка моделей...")
yolo_tree = YOLO(TREE_MODEL_PATH)
yolo_stick = YOLO(STICK_MODEL_PATH)
print("Модели загружены!")


# ======== ENDPOINT: АНАЛИЗ =========
@app.post("/analyze")
async def analyze_tree(
    file: UploadFile = File(...),
    lat: float = Form(0.0),
    lon: float = Form(0.0)
):
    try:
        # Читаем изображение
        image_data = await file.read()
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Пример простого анализа: YOLO для дерева и ствола
        results_tree = yolo_tree(img)
        results_stick = yolo_stick(img)

        # Извлекаем примерные параметры
        height_m = np.random.uniform(5, 20)  # заменить на реальный расчёт
        crown_width_m = np.random.uniform(2, 10)
        trunk_diameter_m = np.random.uniform(0.1, 1.0)

        # Получаем погоду
        weather = get_weather(lat, lon)
        wind_speed = weather.get("wind_speed", 5.0) if isinstance(weather, dict) else 5.0

        # Рассчитываем риск
        risk = calculate_risk(height_m, crown_width_m, trunk_diameter_m, wind_speed)

        return {
            "height_m": round(height_m, 2),
            "crown_width_m": round(crown_width_m, 2),
            "trunk_diameter_m": round(trunk_diameter_m, 2),
            "weather": weather,
            "risk": risk
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ======== ENDPOINT: ЗАГРУЗКА ФОТО ДЛЯ ОБУЧЕНИЯ =========
@app.post("/upload_training_sample")
async def upload_training_sample(file: UploadFile = File(...)):
    """Сохраняет новое изображение в папку для обучения"""
    try:
        os.makedirs(IMAGE_UPLOAD_PATH, exist_ok=True)
        file_path = os.path.join(IMAGE_UPLOAD_PATH, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"status": "uploaded", "filename": file.filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ======== ENDPOINT: АВТООБУЧЕНИЕ =========
@app.post("/train_now")
async def train_now():
    """Запускает дообучение YOLO модели на новых данных"""
    try:
        if not os.path.exists(DATASET_PATH):
            return {"status": "error", "message": "Dataset not found"}

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
