# main.py — обновлённая версия с погодой
import os
import cv2
import numpy as np
import torch
import requests
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image
import io

# =========================
# НАСТРОЙКИ
# =========================
TREE_MODEL_PATH = "tree_model.pt"
STICK_MODEL_PATH = "stick_model.pt"
CLASSIFIER_PATH = "classifier.pth"
CLASS_NAMES_RU = ["берёза", "дуб", "ель", "сосна", "тополь"]
REAL_STICK_LENGTH_M = 1.0
OPENWEATHER_API_KEY = "dc825ffd002731568ec7766eafb54bc9"  # ← ЗАМЕНИТЕ НА СВОЙ!

# =========================
# ЗАГРУЗКА МОДЕЛЕЙ
# =========================
print("Загрузка моделей...")
yolo_tree = YOLO(TREE_MODEL_PATH)
yolo_stick = YOLO(STICK_MODEL_PATH)

clf = None
clf_tfm = None
if os.path.exists(CLASSIFIER_PATH):
    clf = models.resnet18(weights=None)
    clf.fc = torch.nn.Linear(clf.fc.in_features, len(CLASS_NAMES_RU))
    state = torch.load(CLASSIFIER_PATH, map_location="cpu")
    clf.load_state_dict(state, strict=True)
    clf.eval()
    clf_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
print("Модели загружены!")

# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================
def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return None
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    if mask.max() == 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) * 255

def measure_tree(mask: np.ndarray, meters_per_px: float):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or meters_per_px is None:
        return None, None, None
    y_min, y_max = ys.min(), ys.max()
    height_px = y_max - y_min
    height_m = height_px * meters_per_px

    crown_top = int(y_min)
    crown_bot = int(y_min + 0.7 * height_px)
    crown_w = 0
    for y in range(crown_top, crown_bot):
        row = np.where(mask[y] > 0)[0]
        if len(row) > 0:
            crown_w = max(crown_w, row.max() - row.min())
    crown_m = crown_w * meters_per_px

    trunk_top = int(y_max - 0.2 * height_px)
    trunk_w = []
    for y in range(trunk_top, y_max):
        row = np.where(mask[y] > 0)[0]
        if len(row) > 0:
            width = row.max() - row.min()
            if width > 10:
                trunk_w.append(width)

    trunk_m = (np.mean(trunk_w) * meters_per_px) if trunk_w else None

    return round(height_m, 2), round(crown_m, 2), round(trunk_m, 2) if trunk_m else None

def classify_tree(img_bgr, bbox):
    if clf is None:
        return "не определён"
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return "не определён"
    crop = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    crop = Image.fromarray(crop)
    tens = clf_tfm(crop).unsqueeze(0)
    with torch.no_grad():
        logits = clf(tens)
        cls_id = int(torch.argmax(logits, dim=1).item())
    return CLASS_NAMES_RU[cls_id]

def get_weather(lat: float, lon: float):
    """Получает данные о ветре по координатам"""
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url).json()
        wind_speed = response['wind']['speed']  # м/с
        wind_dir = response['wind']['deg']      # градусы
        return wind_speed, wind_dir
    except Exception as e:
        return None, None

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Tree Risk Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_tree(
    file: UploadFile = File(...),
    lat: float = Query(None, description="Широта"),
    lon: float = Query(None, description="Долгота")
):
    try:
        # Чтение изображения
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {"error": "Не удалось загрузить изображение"}

        H, W = img_bgr.shape[:2]

        # ---------- дерево ----------
        res_tree = yolo_tree(img_bgr)[0]
        if res_tree.masks is None or len(res_tree.masks) == 0:
            return {"error": "Дерево не найдено"}

        # Берём самую большую маску
        areas = [cv2.contourArea(cv2.convexHull(m.astype(np.int32))) for m in res_tree.masks.xy]
        idx = int(np.argmax(areas))
        mask = (res_tree.masks.data[idx].cpu().numpy() > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = postprocess_mask(mask)
        xyxy = res_tree.boxes.xyxy[idx].cpu().numpy()

        if mask is None or mask.max() == 0:
            return {"error": "Маска дерева пуста"}

        # ---------- палка ----------
        scale = None
        res_stick = yolo_stick(img_bgr, conf=0.3)[0]
        if len(res_stick.boxes) > 0:
            best_box = max(res_stick.boxes, key=lambda b: (b.xyxy[0][3] - b.xyxy[0][1]))
            x1s, y1s, x2s, y2s = best_box.xyxy[0].cpu().numpy().astype(int)
            stick_height_px = y2s - y1s

            if stick_height_px > 20:
                scale_tmp = REAL_STICK_LENGTH_M / stick_height_px
                if 0.001 < scale_tmp < 0.05:
                    scale = scale_tmp
                    cv2.rectangle(img_bgr, (x1s, y1s), (x2s, y2s), (0, 128, 255), 2)

        # ---------- измерения ----------
        h_m, cw_m, dbh_m = None, None, None
        if mask is not None and mask.max() > 0:
            meters_per_px = scale if scale is not None else 1.0
            h_m, cw_m, dbh_m = measure_tree(mask, meters_per_px)

        # ---------- классификация ----------
        species = classify_tree(img_bgr, xyxy)

        # ---------- погода ----------
        wind_speed = None
        wind_dir = None
        if lat is not None and lon is not None:
            wind_speed, wind_dir = get_weather(lat, lon)

        # ---------- формируем ответ ----------
        result = {
            "success": True,
            "species": species,
            "scale": scale if scale is not None else None,
            "message": f"Масштаб: 1 px = {scale:.4f} м" if scale else "⚠️ Масштаб не найден — размеры в пикселях"
        }

        if h_m is not None:
            result["height_m"] = h_m
            result["crown_width_m"] = cw_m
            if dbh_m is not None:
                result["trunk_diameter_m"] = dbh_m

        if wind_speed is not None:
            result["wind_speed_ms"] = wind_speed
            result["wind_direction_deg"] = wind_dir

        return result

    except Exception as e:
        return {"error": str(e)}