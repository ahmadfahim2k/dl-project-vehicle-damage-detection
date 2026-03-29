import sys
import uuid
import os
sys.path.append('../streamlit-app')

from fastapi import FastAPI, UploadFile, File
from model_helper import predict

TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

app = FastAPI()


@app.get('/hello')
async def hello():
    return "Hello world"

@app.post('/debug')
async def debug_image(file: UploadFile = File(...)):
    from PIL import Image
    import io
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    return {
        'filename': file.filename,
        'content_type': file.content_type,
        'size_bytes': len(image_bytes),
        'image_size': image.size,
        'image_mode': image.mode,
    }

@app.post('/predict')
async def get_prediction(file: UploadFile = File(...)):

    image_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.jpg")
    try:
        image_bytes = await file.read()
        with open(image_path, 'wb') as f:
            f.write(image_bytes)

        prediction = predict(image_path)
        return {
            'prediction': prediction
        }
    except Exception as e:
        return {
            'error': str(e)
        }
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)