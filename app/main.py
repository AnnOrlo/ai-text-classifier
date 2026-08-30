# ============================================
# app/main.py - Точка входа в FastAPI приложение
# ============================================

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

# Импортируем наш классификатор
from app.models.classifier import classifier

# ============================================
# СОЗДАНИЕ ПРИЛОЖЕНИЯ
# ============================================

app = FastAPI(
    title="AI vs Human Text Classifier",
    description="Веб-сервис для определения авторства текста",
    version="1.0.0"
)

# ============================================
# ПОДКЛЮЧЕНИЕ СТАТИЧЕСКИХ ФАЙЛОВ И ШАБЛОНОВ
# ============================================

# Создаём папки для статики, если их нет
for folder in ["static", "static/css", "static/images", "static/js"]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

# Подключаем статику
app.mount("/static", StaticFiles(directory="static"), name="static")

# Подключаем шаблоны
templates = Jinja2Templates(directory="app/templates")

# ============================================
# ЭНДПОИНТЫ
# ============================================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    GET / - Главная страница
    """
    # Получаем информацию о модели для отображения
    model_info = classifier.get_info()
    
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "AI Text Classifier",
            "model_info": model_info
        }
    )


@app.post("/predict")
async def predict_text(text: str):
    """
    POST /predict - Классификация текста
    """
    # Используем реальную модель
    result = classifier.predict(text)
    
    # Если есть ошибка — возвращаем 400
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    """
    POST /predict/file - Классификация из файла
    """
    try:
        # Проверяем тип файла
        allowed_extensions = ['.txt', '.docx']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Поддерживаются только файлы: {', '.join(allowed_extensions)}"
            )
        
        # Читаем содержимое файла
        if file_ext == '.txt':
            contents = await file.read()
            try:
                text = contents.decode('utf-8')
            except UnicodeDecodeError:
                text = contents.decode('cp1251', errors='ignore')
        elif file_ext == '.docx':
            # Для .docx нужна библиотека python-docx
            try:
                from docx import Document
                import io
                
                contents = await file.read()
                doc = Document(io.BytesIO(contents))
                text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="Для поддержки .docx установите: pip install python-docx"
                )
        
        # Классифицируем текст
        result = classifier.predict(text)
        
        # Добавляем информацию о файле
        result["filename"] = file.filename
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке файла: {str(e)}"
        )


@app.get("/features")
async def get_features():
    """
    GET /features - Список используемых признаков
    """
    if not classifier.is_loaded:
        return {
            "features": [],
            "total_features": 0,
            "message": "Модель не загружена"
        }
    
    # Получаем топ признаков из модели
    top_features = classifier.get_feature_importance(top_n=50)
    
    # Базовые признаки + топ слова
    base_features = [
        "text_length (длина текста)",
        "word_count (количество слов)",
        "CountVectorizer (5000 признаков)"
    ]
    
    # Топ слова как признаки
    top_words = [f"word: {f['feature']} (важность: {f['importance']:.2%})" 
                 for f in top_features[:20]]
    
    return {
        "features": base_features + top_words,
        "total_features": len(classifier.feature_names),
        "vectorizer_type": "CountVectorizer",
        "ngram_range": "1-2 (униграммы и биграммы)"
    }


@app.get("/feature/importance")
async def get_feature_importance():
    """
    GET /feature/importance - Важность признаков
    """
    if not classifier.is_loaded:
        return {
            "message": "Модель не загружена",
            "top_features": []
        }
    
    # Получаем топ-20 важных признаков из модели
    top_features = classifier.get_feature_importance(top_n=20)
    
    return {
        "message": "Важность признаков модели (нормализована к максимуму)",
        "model_name": classifier.model_info.get('model_name', 'Unknown'),
        "model_accuracy": classifier.model_info.get('accuracy', 0),
        "top_features": top_features
    }


@app.get("/model/info")
async def get_model_info():
    """
    GET /model/info - Информация о модели
    """
    return classifier.get_info()


# ============================================
# ЗАПУСК СЕРВЕРА
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )