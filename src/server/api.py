from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np

# Импорт модели и device
from src.models.megnet import MegaNet, MegaNetInference

app = FastAPI(title="MegaNet Inference API")

# --- Настройка и загрузка модели ---
# Здесь нужно указать путь к файлу весов внутри контейнера:
WEIGHTS_PATH = "src/models/meganet_model_best.pth"

# Гиперпараметры (те же, что при обучении)
best_params = {
    'shared_embed_dim': 64,
    'latent_dim':       32,
    'num_heads':        8,
    'dropout':          0.1865797171518085
}

input_size = 8
num_classes = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Создаём и загружаем модель
base_model = MegaNet(input_size, num_classes, best_params)
base_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
infer_model = MegaNetInference(base_model).to(device)
infer_model.eval()


# --- Схемы Pydantic ---
class Inp(BaseModel):
    values: list[float]  # длина списка должна быть = input_size

class Prediction(BaseModel):
    probabilities: list[float]
    predicted_class: int


# --- Эндпоинт /predict ---
@app.post("/predict", response_model=Prediction)
async def predict(inp: Inp):
    if len(inp.values) != input_size:
        raise HTTPException(status_code=400,
            detail=f"Input vector must have length {input_size}")

    # Подготовка тензора
    x = torch.tensor([inp.values], dtype=torch.float32, device=device)

    # Инференс
    with torch.no_grad():
        logits = infer_model(x)
        # logits: shape [1, num_classes]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
        pred = int(np.argmax(probs))

    return Prediction(probabilities=probs, predicted_class=pred)
