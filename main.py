from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import numpy as np
import cv2
import os
import json
import requests

app = FastAPI(title="🌲 Tree Risk Analyzer — Smart Forestry AI")

# ========== Настройки ==========
TREE_MODEL_PATH = "tree_model.pt"
STICK_MODEL_PATH = "stick_model.pt"
DATASET_PATH = "datasets"
IMAGE_UPLOAD_PATH = os.path.join(DATASET_PATH, "uploads")
os.makedirs(IMAGE_UPLOAD_PATH, exist_ok=True)

# ========== API ключи ==========
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "")
DRIVE_FOLDER_NAME = os.getenv("DRIVE_FOLDER_NAME", "TreeAppUploads")
DRIVE_SHARE_WITH = os.getenv("DRIVE_SHARE_WITH", "")

# ========== Инициализация моделей ==========
print("📦 Загрузка моделей YOLO...")
yolo_tree = YOLO(TREE_MODEL_PATH)
yolo_stick = YOLO(STICK_MODEL_PATH)
print("✅ Модели загружены успешно!")


# ========== Google Drive ==========

def get_drive_service():
    """Создание Google Drive service через ENV JSON"""
    info = json.loads(GOOGLE_CREDENTIALS_JSON)
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=['https://www.googleapis.com/auth/drive.file'])
    return build('drive', 'v3', credentials=creds)


def ensure_drive_folder(service):
    """Создаёт папку на Google Drive (если её нет) и расшаривает на указанный Gmail"""
    q = f"name='{DRIVE_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    res = service.files().list(q=q, fields='files(id,name)').execute()
    files = res.get('files', [])
    if files:
        return files[0]['id']

    meta = {'name': DRIVE_FOLDER_NAME, 'mimeType': 'application/vnd.google-apps.folder'}
    folder = service.files().create(body=meta, fields='id').execute()
    folder_id = folder['id']

    if DRIVE_SHARE_WITH:
        perm = {'type': 'user', 'role': 'writer', 'emailAddress': DRIVE_SHARE_WITH}
        service.permissions().create(
            fileId=folder_id, body=perm, fields='id', sendNotificationEmail=False).execute()

    return folder_id


def upload_to_drive(file_path, filename):
    """Загрузка файла на Google Drive"""
    try:
        service = get_drive_service()
        folder_id = ensure_drive_folder(service)
        metadata = {'name': filename, 'parents': [folder_id]}
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().create(
            body=metadata, media_body=media, fields='id,webViewLink').execute()
        return {"status": "uploaded", "file_id": file.get('id'), "webViewLink": file.get('webViewLink')}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ========== Погода ==========

def get_weather(lat, lon):
    """Получение скорости и направления ветра"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url).json()
        return {
            "wind_speed": res["wind"]["speed"],
            "wind_deg": res["wind"]["deg"],
            "temp": res["main"]["temp"],
            "desc": res["weather"][0]["description"]
        }
    except Exception as e:
        return {"error": str(e)}


# ========== Расчёт риска ==========
def calculate_risk(height_m, crown_width_m, trunk_diameter_m, wind_speed):
    if height_m == 0 or crown_width_m == 0:
        return {"risk": 0, "category": "No tree detected"}
    ratio = crown_width_m / max(trunk_diameter_m, 0.1)
    wind_factor = wind_speed / 10
    risk = min(ratio * wind_factor * (height_m / 10), 1.0)
    category = "Low" if risk < 0.3 else "Medium" if risk < 0.7 else "High"
    return {"risk": round(risk, 2), "category": category}


# ========== Основной анализ ==========
@app.post("/analyze")
async def analyze_tree(
    file: UploadFile = File(...),
    lat: float = Form(0.0),
    lon: float = Form(0.0),
    stick_length_m: float = Form(1.0),
    manual_stick: str = Form(None)
):
    try:
        # === Читаем изображение ===
        image_data = await file.read()
        np_img = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        local_path = os.path.join(IMAGE_UPLOAD_PATH, file.filename)
        cv2.imwrite(local_path, img)

        # === Поиск палки ===
        results_stick = yolo_stick(img)
        stick_boxes = results_stick[0].boxes.xyxy.cpu().numpy()
        stick_status = ""
        scale_m_per_px = None

        if len(stick_boxes) > 0:
            x1, y1, x2, y2 = stick_boxes[0]
            stick_length_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            scale_m_per_px = stick_length_m / stick_length_px
            stick_status = "✅ Палка найдена, масштаб вычислен."
        elif manual_stick:
            pts = manual_stick.split(",")
            x1, y1, x2, y2 = map(float, pts)
            stick_length_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            scale_m_per_px = stick_length_m / stick_length_px
            stick_status = "✋ Палка отмечена вручную."
        else:
            stick_status = "⚠️ Палка не найдена — масштаб вычислить невозможно."

        # === Анализ дерева ===
        results_tree = yolo_tree(img)
        boxes = results_tree[0].boxes.xyxy.cpu().numpy()

        if len(boxes) > 0 and scale_m_per_px:
            x1, y1, x2, y2 = boxes[0]
            height_px = y2 - y1
            crown_width_px = x2 - x1
            height_m = height_px * scale_m_per_px
            crown_width_m = crown_width_px * scale_m_per_px
            trunk_diameter_m = 0.1 * height_m
        else:
            height_m = crown_width_m = trunk_diameter_m = 0

        # === Погода ===
        weather = get_weather(lat, lon)
        wind_speed = weather.get("wind_speed", 5.0)

        # === Риск ===
        risk = calculate_risk(height_m, crown_width_m, trunk_diameter_m, wind_speed)

        # === Загрузка в Google Drive ===
        drive_upload = upload_to_drive(local_path, file.filename)

        return {
            "stick_status": stick_status,
            "height_m": round(height_m, 2),
            "crown_width_m": round(crown_width_m, 2),
            "trunk_diameter_m": round(trunk_diameter_m, 2),
            "risk": risk,
            "weather": weather,
            "cloud_upload": drive_upload
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
