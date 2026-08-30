import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class TextClassifier:
    """
    Класс для работы с моделью классификации текстов.
    Загружает модель и векторизатор, выполняет предсказания.
    """
    
    def __init__(self, model_path: str = "saved_models/best_model_xgboost.pkl"):

        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.feature_names = []
        self.model_info = {}
        self.is_loaded = False
        
        # Загружаем модель при создании объекта
        self.load_model()
    
    def load_model(self) -> bool:
        try:
            # Проверяем существование файла
            if not Path(self.model_path).exists():
                print(f"Файл модели не найден: {self.model_path}")
                return False
            
            # Загружаем пакет с моделью
            package = joblib.load(self.model_path)
            
            # Извлекаем компоненты
            self.model = package['model']
            self.vectorizer = package['vectorizer']
            self.model_info = {
                'model_name': package.get('model_name', 'Unknown'),
                'accuracy': package.get('accuracy', 0),
                'f1': package.get('f1', 0),
                'best_params': package.get('best_params', {}),
                'creation_time': package.get('creation_time', '')
            }
            
            # Получаем названия признаков из векторизатора
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            self.is_loaded = True
            print(f"Модель загружена успешно:")
            print(f"   Название: {self.model_info['model_name']}")
            print(f"   Accuracy: {self.model_info['accuracy']:.4f}")
            print(f"   Признаков: {len(self.feature_names)}")
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, text: str) -> Dict:
        """
        Классифицирует текст и возвращает результат.
        """
        if not self.is_loaded:
            return {
                "class": "unknown",
                "confidence": 0.0,
                "error": "Модель не загружена"
            }
        
        try:
            # Валидация входного текста
            if not text or len(text.strip()) < 10:
                return {
                    "class": "unknown",
                    "confidence": 0.0,
                    "error": "Текст слишком короткий (минимум 10 символов)"
                }
            
            # Определяем язык
            language = self.detect_language(text)
            
            # Векторизуем текст
            text_vectorized = self.vectorizer.transform([text])
            text_vectorized = text_vectorized.astype(np.float32)
            
            # Получаем предсказание
            prediction = self.model.predict(text_vectorized)[0]
            probabilities = self.model.predict_proba(text_vectorized)[0]
            
            class_label = int(prediction)
            confidence = float(probabilities[class_label])
            
            class_names = {
                0: "Human-written",
                1: "AI-generated"
            }
            
            # Формируем результат
            result = {
                "class": class_names.get(class_label, "unknown"),
                "class_label": class_label,
                "confidence": round(confidence, 4),
                "probabilities": {
                    "human": round(float(probabilities[0]), 4),
                    "ai": round(float(probabilities[1]), 4)
                },
                "language": language,
                "text_length": len(text),
                "word_count": len(text.split())
            }
            
            # Добавляем предупреждение для не-английских текстов
            if language != 'en':
                result["warning"] = (
                    "Модель обучена на английском языке. "
                    "Для текстов на других языках результат может быть неточным."
                )
            
            return result
            
        except Exception as e:
            return {
                "class": "unknown",
                "confidence": 0.0,
                "error": f"Ошибка при предсказании: {str(e)}"
            }
    
    def get_feature_importance(self, top_n: int = 20) -> List[Dict]:

        if not self.is_loaded:
            return []
        
        try:
            # Получаем важность признаков из модели
            importances = self.model.feature_importances_
            
            # Сортируем по убыванию важности
            top_indices = np.argsort(importances)[::-1][:top_n]
            
            # Формируем список результатов
            top_features = []
            max_importance = importances[top_indices[0]]
            
            for idx in top_indices:
                top_features.append({
                    "feature": self.feature_names[idx],
                    "importance": round(float(importances[idx] / max_importance), 4),
                    "raw_importance": round(float(importances[idx]), 6)
                })
            
            return top_features
            
        except Exception as e:
            print(f"Ошибка при получении важности признаков: {e}")
            return []
    
    def get_info(self) -> Dict:
        """Возвращает информацию о модели."""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        return {
            **self.model_info,
            "n_features": len(self.feature_names),
            "model_type": type(self.model).__name__,
            "vectorizer_type": type(self.vectorizer).__name__
        }

    #Определение языка (из-за того что датасет работает преимущественно на английском)
    def detect_language(self, text: str) -> str:
        cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        latin_count = sum(1 for c in text if '\u0041' <= c <= '\u007A')
        
        if cyrillic_count > latin_count:
            return 'ru'
        elif latin_count > cyrillic_count:
            return 'en'
        else:
            return 'unknown'

# Создаём один экземпляр на всё приложение
classifier = TextClassifier()