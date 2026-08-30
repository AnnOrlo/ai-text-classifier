# AI vs Human Text Classifier

Веб-сервис для классификации текстов: написан человеком или сгенерирован ИИ.

## Требования

- Python 3.10+
- pip
- git

## Установка

    git clone https://github.com/AnnOrlo/ai-text-classifier.git
    cd ai-text-classifier
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## Запуск

    python -m app.main

Сервер доступен по адресу:

    http://127.0.0.1:8000

## Структура проекта

    ai-text-classifier/
    ├── app/
    │   ├── main.py
    │   ├── models/
    │   │   └── classifier.py
    │   └── templates/
    │       └── index.html
    ├── saved_models/
    │   └── best_model_xgboost.pkl
    ├── notebooks/
    │   └── research.ipynb
    ├── scripts/
    │   └── retrain_model.py
    ├── static/
    │   └── css/
    │       └── style.css
    ├── requirements.txt
    └── README.md

## API

| Метод | Эндпоинт | Описание |
|---|---|---|
| GET | `/` | Главная страница |
| POST | `/predict` | Классификация текста |
| POST | `/predict/file` | Классификация из файла `.txt` или `.docx` |
| GET | `/features` | Список признаков |
| GET | `/feature/importance` | Важность признаков |

## Пример запроса

    POST /predict?text=Your text here

## Пример ответа

    {
      "class": "AI-generated",
      "class_label": 1,
      "confidence": 0.9876,
      "probabilities": {
        "human": 0.0124,
        "ai": 0.9876
      },
      "language": "en",
      "text_length": 234,
      "word_count": 45
    }

## Модель

- Алгоритм: XGBoost
- Векторизация: CountVectorizer
- Количество признаков: 5000
- N-grams: 1-2
- Датасет: AI vs Human Text, Kaggle

## Метрики на тестовой выборке

| Метрика | Значение |
|---|---|
| Accuracy | 0.9946 |
| Precision | 0.9956 |
| Recall | 0.9898 |
| F1-score | 0.9927 |

## Гиперпараметры модели

| Параметр | Значение |
|---|---|
| n_estimators | 200 |
| learning_rate | 0.2 |
| max_depth | 7 |
| subsample | 0.8 |

## Ограничения

- Модель обучена на англоязычных школьных эссе.
- Для текстов менее 50 слов результат может быть неточным.
- Тексты на других языках не поддерживаются.
- Модель лучше всего работает со структурированными текстами на английском языке.

## Переобучение модели

Для переобучения модели необходимо поместить датасет в папку:

    data/generated_ai_texts.csv

После этого запустить:

    python scripts/retrain_model.py

## Примечание по данным

Исходный датасет не загружается в репозиторий и исключён через `.gitignore`.

Для повторного обучения его необходимо скачать отдельно с Kaggle:

https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text