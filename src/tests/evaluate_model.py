import pandas as pd
import requests
import numpy as np
from sklearn.metrics import accuracy_score

# Путь к датасету
DATASET_PATH = "/home/user/megnet-deployment/src/models/filled_dataset_optimized_new_era.csv"

# URL эндпоинта API
API_URL = "http://localhost:8080/predict_ensemble"

# Словарь с названиями столбцов
col_dict = {
    34: "ВОЗРАСТ.НЕ.ПОМЕХА.количество.баллов.(СКРИНИНГ.«ВОЗРАСТ.НЕ.ПОМЕХА»)",
    78: "Физическая.активность.-.кратность.(Факторы.риска.хронических.неинфекционных.заболеваний)",
    79: "Физическая.активность.-.продолжительность.(Факторы.риска.хронических.неинфекционных.заболеваний)",
    105: "ИМТ.(кг/м^2).(Осмотр)",
    347: "индекс.Бартел.количество.баллов.(Шкала.базовой.активности.в.повседневной.жизни.(индекс.Бартел).-ADL)",
    367: "SPPB.количество.баллов.(Кратка)",
    378: "Тест.«встань.и.иди».(сек).(≤10.–.норма;.≥14.-.риск.падений):.(Тест.«встань.и.иди»)",
    363: "Ходьба.на.4.м.(Время,.секунды).(Кратка)",
    380: "Уровень.(Стратификация.по.уровню.физической.активности)"
}

# Определяем столбцы-признаки и столбец-метку
features = [col_dict[key] for key in col_dict if key != 380]
label = col_dict[380]

# Загружаем датасет
df = pd.read_csv(DATASET_PATH)

# Проверяем наличие всех необходимых столбцов
missing_cols = [col for col in features + [label] if col not in df.columns]
if missing_cols:
    raise ValueError(f"В датасете отсутствуют столбцы: {missing_cols}")

# Извлекаем признаки и метки
X = df[features].values
y_true = df[label].values

# Список для хранения предсказанных меток
y_pred = []

# Отправляем запросы для каждого примера
for i, x in enumerate(X):
    # Формируем JSON-запрос
    payload = {"values": x.tolist()}
    
    try:
        # Отправляем POST-запрос к эндпоинту
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Проверяем, успешен ли запрос
        
        # Извлекаем предсказанную метку
        pred_class = response.json().get("predicted_class")
        y_pred.append(pred_class)
        
        # Для примера из запроса проверяем конкретный случай
        if np.allclose(x, [2.0, 4.0, 2.0, 38.06, 100.0, 11.0, 7.61, 4.0]):
            print(f"Пример из запроса: {payload['values']}")
            print(f"Предсказанная метка: {pred_class}")
            print(f"Реальная метка: {y_true[i]}")
            print(f"Совпадение: {pred_class == y_true[i]}")
        
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при обработке примера {i}: {e}")
        y_pred.append(None)  # Добавляем None для пропущенных предсказаний

# Удаляем None из предсказаний и соответствующие метки
valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
y_pred_valid = [y_pred[i] for i in valid_indices]
y_true_valid = [y_true[i] for i in valid_indices]

# Вычисляем точность
if valid_indices:
    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    print(f"\nТочность модели: {accuracy:.4f} ({len(valid_indices)} примеров обработано)")
else:
    print("Не удалось обработать ни один пример")