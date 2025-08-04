FROM python:3.10-slim

# Устанавливаем зависимости ОС
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код
COPY src/ ./src/

EXPOSE 8080

# Запуск FastAPI
CMD ["uvicorn", "src.server.api:app", "--host", "0.0.0.0", "--port", "8080"]
