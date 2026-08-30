# ============================================
# ИСПРАВЛЕНИЕ SSL ПРОБЛЕМЫ
# ============================================

import ssl
import certifi
import os

# Устанавливаем правильные сертификаты
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Альтернатива: отключаем проверку SSL (если не помогло выше)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("✓ SSL исправления применены")

# Теперь обычные импорты
import pandas as pd
import numpy as np
# ... и так далее
# retrain_model.py - Пересоздание XGBoost модели локально. Понадобилось из-за разностй версий библиотек в colab и локально

import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier


# 1. ЗАГРУЗКА ДАННЫХ (ИЗ ЛОКАЛЬНОГО ФАЙЛА) - т.к. возникли проблемы с ssl при удаленном запросе

print("Загрузка данных из локального файла...")

# Ищем CSV файл в папке data
csv_path = None
possible_paths = [
    'data/generated_ai_texts.csv',
    'data/ai-vs-human-text.csv',
    'data/generated_texts.csv',
]

for path in possible_paths:
    if os.path.exists(path):
        csv_path = path
        break

# Если не нашли в data/, ищем в корне
if not csv_path:
    for f in os.listdir('.'):
        if f.endswith('.csv') and ('text' in f.lower() or 'generated' in f.lower() or 'ai' in f.lower()):
            csv_path = f
            break

if not csv_path:
    print("❌ CSV файл не найден!")
    print("   Пожалуйста, скачай данные с Kaggle и положи CSV в папку data/")
    print("   Ссылка: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text")
    exit(1)

print(f"✓ Найден файл: {csv_path}")

# Загружаем данные
df = pd.read_csv(csv_path)
print(f"✓ Загружено {len(df):,} записей")
print(f"✓ Столбцы: {list(df.columns)}")

# Определяем целевой столбец
target_col = None
for col in ['generated', 'label', 'is_ai', 'target']:
    if col in df.columns:
        target_col = col
        break

if not target_col:
    print(f"Не найден целевой столбец! Доступные: {list(df.columns)}")
    exit(1)

# Определяем столбец с текстом
text_col = None
for col in ['text', 'content', 'prompt']:
    if col in df.columns:
        text_col = col
        break

if not text_col:
    print(f"Не найден столбец с текстом!")
    exit(1)

print(f"✓ Целевой столбец: {target_col}")
print(f"✓ Текстовый столбец: {text_col}")

# 2. ПОДГОТОВКА ДАННЫХ
# Уменьшили выборку без потери качества, но с увеличением скорости

SAMPLE_SIZE = 80000
if len(df) > SAMPLE_SIZE:
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"Используем выборку {SAMPLE_SIZE:,} записей")
else:
    df_sample = df
    print(f"Используем все {len(df):,} записей")

y = df_sample[target_col].values.astype(int)
print(f"Баланс классов: {np.bincount(y)}")

# 3. ВЕКТОРИЗАЦИЯ

print("Векторизация текстов...")
vectorizer = CountVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    stop_words='english'
)

start_time = time.time()
X = vectorizer.fit_transform(df_sample[text_col])
print(f"Векторизация завершена за {time.time() - start_time:.2f} сек")
print(f"Размер матрицы: {X.shape}")
print(f"Количество признаков: {len(vectorizer.get_feature_names_out()):,}")


# 4. РАЗДЕЛЕНИЕ НА TRAIN/TEST


print("Разделение данных...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

print(f"Train: {X_train.shape[0]:,} записей")
print(f"Test: {X_test.shape[0]:,} записей")

# 5. ОБУЧЕНИЕ XGBOOST

print("Обучение XGBoost...")
print("Параметры (из тюнинга в Colab):")
print("   n_estimators: 200")
print("   learning_rate: 0.2")
print("   max_depth: 7")
print("   subsample: 0.8")

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.2,
    max_depth=7,
    subsample=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    verbosity=0
)

start_time = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_time

print(f" завершено за {train_time:.2f} сек")

# 6. ОЦЕНКА МОДЕЛИ

print("Оценка модели на тестовой выборке...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-score:  {f1:.4f}")

# 7. СОХРАНЕНИЕ МОДЕЛИ

print("Сохранение модели...")
os.makedirs('saved_models', exist_ok=True)

model_package = {
    'model': model,
    'vectorizer': vectorizer,
    'model_name': 'XGBoost (Locally Trained)',
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'best_params': {
        'n_estimators': 200,
        'learning_rate': 0.2,
        'max_depth': 7,
        'subsample': 0.8
    },
    'n_features': len(vectorizer.get_feature_names_out()),
    'sample_size': SAMPLE_SIZE,
    'creation_time': pd.Timestamp.now().isoformat()
}

joblib.dump(model_package, 'saved_models/best_model_xgboost.pkl', protocol=4)

file_size = os.path.getsize('saved_models/best_model_xgboost.pkl') / (1024 * 1024)
print(f"Модель сохранена: saved_models/best_model_xgboost.pkl")
print(f"Размер файла: {file_size:.2f} МБ")

# 8. ПРОВЕРКА ЗАГРУЗКИ

print("Проверка загрузки модели...")
test_package = joblib.load('saved_models/best_model_xgboost.pkl')
print(f"Ключи: {list(test_package.keys())}")
print(f"Тип модели: {type(test_package['model']).__name__}")
print(f"Тип векторизатора: {type(test_package['vectorizer']).__name__}")

# Тест предсказания
test_text = "Artificial intelligence has revolutionized the way we approach complex problems."
test_vec = test_package['vectorizer'].transform([test_text]).astype(np.float32)
test_pred = test_package['model'].predict(test_vec)[0]
test_proba = test_package['model'].predict_proba(test_vec)[0]

print(f"Тест предсказания:")
print(f"Текст: {test_text[:50]}...")
print(f"Класс: {test_pred} ({'AI' if test_pred == 1 else 'Human'})")
print(f"Уверенность: {test_proba[test_pred]*100:.2f}%")