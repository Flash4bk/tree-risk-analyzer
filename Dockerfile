FROM python:3.10-slim

# Устанавливаем системные зависимости для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем код
WORKDIR /app
COPY . .

# Устанавливаем зависимости Python (включая Torch CPU)
RUN pip install --no-cache-dir -r requirements.txt

# Запускаем сервер
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
