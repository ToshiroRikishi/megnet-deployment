FROM python:3.10-slim

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Устанавливаем зависимости ОС
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создаем непривилегированного пользователя
RUN useradd -m -u 1000 appuser

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код и существующие модели
COPY src/ ./src/
COPY src/models/MegaNet_model_classes_1_3_vs_4_5.pth ./src/models/
COPY src/models/MegaNet_model_classes_1_2_3.pth ./src/models/
COPY src/models/MegaNet_model_classes_4_5.pth ./src/models/

# Переключаемся на непривилегированного пользователя
USER appuser

# Открываем порт
EXPOSE 8080

# Запуск FastAPI с несколькими рабочими процессами для продакшн
CMD ["uvicorn", "src.server.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]