from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

# Импорт модели и device
from src.models.megnet import MegaNet, MegaNetInference

app = FastAPI(title="MegaNet Inference API")

# --- Настройка и загрузка одиночной модели ---
WEIGHTS_PATH = "src/models/meganet_model_best.pth"

# Гиперпараметры (те же, что при обучении)
best_params = {
    'shared_embed_dim': 64,
    'latent_dim': 32,
    'num_heads': 8,
    'dropout': 0.1865797171518085
}

input_size = 8
num_classes = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Создаём и загружаем одиночную модель
base_model = MegaNet(input_size, num_classes, best_params)
checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)

# Обработка разных форматов сохранения модели
if 'model_state_dict' in checkpoint:
    base_model.load_state_dict(checkpoint['model_state_dict'])
    scaler = checkpoint['scaler']
elif 'state_dict' in checkpoint:
    base_model.load_state_dict(checkpoint['state_dict'])
    scaler = checkpoint['scaler']
else:
    # Если это просто state_dict без обертки
    if 'scaler' in checkpoint:
        scaler = checkpoint['scaler']
        # Удаляем scaler из checkpoint и загружаем остальное как state_dict
        model_state = {k: v for k, v in checkpoint.items() if k != 'scaler'}
        base_model.load_state_dict(model_state)
    else:
        # Если это только веса модели без скейлера
        base_model.load_state_dict(checkpoint)
        # Создаем dummy скейлер (это не идеально, но позволит запустить)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Устанавливаем некоторые разумные значения для скейлера
        scaler.mean_ = np.zeros(input_size)
        scaler.scale_ = np.ones(input_size)

infer_model = MegaNetInference(base_model).to(device)
infer_model.eval()

# --- Настройка и загрузка ансамбля моделей ---
# Предопределенные гиперпараметры для ансамбля MegaNet
model_params = {
    'binary_1_3_vs_4_5': {'lr': 0.003261994572554154, 'dropout': 0.20634953989793595, 'shared_embed_dim': 128, 'latent_dim': 128, 'num_heads': 8, 'recon_weight': 0.13003271742797662, 'kl_weight': 0.007706404963519681, 'batch_size': 128},
    'multiclass_1_2_3': {'lr': 0.0001652456054246614, 'dropout': 0.1766769260535775, 'shared_embed_dim': 256, 'latent_dim': 128, 'num_heads': 4, 'recon_weight': 0.26985140479536573, 'kl_weight': 0.00038882113500409305, 'batch_size': 64},
    'binary_4_5': {'lr': 0.00018725722022078217, 'dropout': 0.2417088813617859, 'shared_embed_dim': 64, 'latent_dim': 128, 'num_heads': 8, 'recon_weight': 0.053753726714484695, 'kl_weight': 0.003242457328313236, 'batch_size': 64}
}

# Пути к сохраненным моделям ансамбля
model_paths = {
    'binary_1_3_vs_4_5': 'src/models/MegaNet_model_classes_1_3_vs_4_5.pth',
    'multiclass_1_2_3': 'src/models/MegaNet_model_classes_1_2_3.pth',
    'binary_4_5': 'src/models/MegaNet_model_classes_4_5.pth'
}

def load_meganet_model(model_path, input_size, num_classes, params, device):
    """Загружает обученную модель MegaNet"""
    model = MegaNet(input_size=input_size, num_classes=num_classes, params=params).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Обработка разных форматов сохранения модели
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = checkpoint['scaler']
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        scaler = checkpoint['scaler']
    else:
        # Если это просто state_dict без обертки
        if 'scaler' in checkpoint:
            scaler = checkpoint['scaler']
            # Удаляем scaler из checkpoint и загружаем остальное как state_dict
            model_state = {k: v for k, v in checkpoint.items() if k != 'scaler'}
            model.load_state_dict(model_state)
        else:
            # Если это только веса модели без скейлера
            model.load_state_dict(checkpoint)
            # Создаем dummy скейлер
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.mean_ = np.zeros(input_size)
            scaler.scale_ = np.ones(input_size)
    
    model.eval()
    return model, scaler

# Параметры для каждой модели ансамбля
ensemble_configs = {
    'binary_1_3_vs_4_5': {'input_size': 8, 'num_classes': 2},
    'multiclass_1_2_3': {'input_size': 8, 'num_classes': 3},
    'binary_4_5': {'input_size': 8, 'num_classes': 2}
}

# Загрузка ансамбля моделей
print("Загрузка ансамбля моделей...")
best_params_1_3_vs_4_5 = model_params['binary_1_3_vs_4_5']
model_1_3_vs_4_5, scaler_1_3_vs_4_5 = load_meganet_model(
    model_paths['binary_1_3_vs_4_5'], 
    ensemble_configs['binary_1_3_vs_4_5']['input_size'], 
    ensemble_configs['binary_1_3_vs_4_5']['num_classes'], 
    best_params_1_3_vs_4_5, 
    device
)

best_params_1_2_3 = model_params['multiclass_1_2_3']
model_1_2_3, scaler_1_2_3 = load_meganet_model(
    model_paths['multiclass_1_2_3'], 
    ensemble_configs['multiclass_1_2_3']['input_size'], 
    ensemble_configs['multiclass_1_2_3']['num_classes'], 
    best_params_1_2_3, 
    device
)

best_params_4_5 = model_params['binary_4_5']
model_4_5, scaler_4_5 = load_meganet_model(
    model_paths['binary_4_5'], 
    ensemble_configs['binary_4_5']['input_size'], 
    ensemble_configs['binary_4_5']['num_classes'], 
    best_params_4_5, 
    device
)

# Обертка для ансамбля моделей
class MegaNetEnsembleInference:
    def __init__(self, model_1_3_vs_4_5, scaler_1_3_vs_4_5, model_1_2_3, scaler_1_2_3, model_4_5, scaler_4_5, device):
        self.model_1_3_vs_4_5 = model_1_3_vs_4_5
        self.scaler_1_3_vs_4_5 = scaler_1_3_vs_4_5
        self.model_1_2_3 = model_1_2_3
        self.scaler_1_2_3 = scaler_1_2_3
        self.model_4_5 = model_4_5
        self.scaler_4_5 = scaler_4_5
        self.device = device
        
        # Переводим все модели в режим inference
        for model in [self.model_1_3_vs_4_5, self.model_1_2_3, self.model_4_5]:
            model.eval()
            # Отключаем dropout
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = 0.0
    
    def predict(self, X):
        # Масштабирование данных для первой модели
        X_scaled_1_3_vs_4_5 = self.scaler_1_3_vs_4_5.transform(X.reshape(1, -1))
        X_tensor_1_3_vs_4_5 = torch.FloatTensor(X_scaled_1_3_vs_4_5).to(self.device)
        
        # Предсказание первой модели (1-3 vs 4-5)
        with torch.no_grad():
            outputs_1_3_vs_4_5, _, _, _ = self.model_1_3_vs_4_5(X_tensor_1_3_vs_4_5)
            probs_1_3_vs_4_5 = torch.sigmoid(outputs_1_3_vs_4_5).cpu().numpy()
            initial_pred = (probs_1_3_vs_4_5 > 0.5).astype(int)[0]
        
        # Если предсказание 0 (классы 1-3)
        if initial_pred == 0:
            X_scaled_1_2_3 = self.scaler_1_2_3.transform(X.reshape(1, -1))
            X_tensor_1_2_3 = torch.FloatTensor(X_scaled_1_2_3).to(self.device)
            
            with torch.no_grad():
                outputs_1_2_3, _, _, _ = self.model_1_2_3(X_tensor_1_2_3)
                pred_1_2_3 = torch.argmax(outputs_1_2_3, dim=1).cpu().numpy()[0]
                final_pred = pred_1_2_3 + 1  # Сдвиг обратно к 1, 2, 3
        
        # Если предсказание 1 (классы 4-5)
        else:
            X_scaled_4_5 = self.scaler_4_5.transform(X.reshape(1, -1))
            X_tensor_4_5 = torch.FloatTensor(X_scaled_4_5).to(self.device)
            
            with torch.no_grad():
                outputs_4_5, _, _, _ = self.model_4_5(X_tensor_4_5)
                probs_4_5_raw = torch.sigmoid(outputs_4_5).cpu().numpy()
                pred_4_5 = (probs_4_5_raw > 0.5).astype(int)[0]
                final_pred = pred_4_5 + 4  # Сдвиг к 4 или 5
        
        return final_pred

# Создаем экземпляр ансамбля
ensemble_model = MegaNetEnsembleInference(
    model_1_3_vs_4_5, scaler_1_3_vs_4_5,
    model_1_2_3, scaler_1_2_3,
    model_4_5, scaler_4_5,
    device
)

# --- Схемы Pydantic ---
class Inp(BaseModel):
    values: list[float]  # длина списка должна быть = input_size

class Prediction(BaseModel):
    probabilities: list[float]
    predicted_class: int

class EnsemblePrediction(BaseModel):
    predicted_class: int

# --- Эндпоинт /predict для одиночной модели ---
@app.post("/predict", response_model=Prediction)
async def predict(inp: Inp):
    if len(inp.values) != input_size:
        raise HTTPException(status_code=400,
                          detail=f"Input vector must have length {input_size}")
    
    # Преобразуем в numpy array
    x_raw = np.array(inp.values)
    
    # Масштабируем данные с помощью загруженного скейлера
    x_scaled = scaler.transform(x_raw.reshape(1, -1))
    
    # Подготовка тензора
    x = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    
    # Инференс
    with torch.no_grad():
        logits = infer_model(x)
        # logits: shape [1, num_classes]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
        pred = int(np.argmax(probs))
    
    return Prediction(probabilities=probs, predicted_class=pred)

# --- Эндпоинт /predict_ensemble для ансамбля моделей ---
@app.post("/predict_ensemble", response_model=EnsemblePrediction)
async def predict_ensemble(inp: Inp):
    if len(inp.values) != input_size:
        raise HTTPException(status_code=400,
                          detail=f"Input vector must have length {input_size}")
    
    # Преобразуем в numpy array
    x_raw = np.array(inp.values)
    
    # Используем ансамбль для предсказания
    predicted_class = ensemble_model.predict(x_raw)
    
    return EnsemblePrediction(predicted_class=int(predicted_class))