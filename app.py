from dotenv import load_dotenv
import os
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from inference import model, predict, preprocess_image, val_transform
from io import BytesIO
from PIL import Image
from fastapi.responses import FileResponse

import time
from datetime import datetime 
from cloud_storage import upload_image
import asyncio

app = FastAPI()

SAVE_DIR = "saved_moments"
os.makedirs(SAVE_DIR, exist_ok=True) 

confidenceThreshold = 0.90
cooldown = 5
lastSaveTime = time.time()
saveLock = asyncio.Lock()

emotions = ["Angry","Disgust","Fear","Sad","Surprise"]

@app.get("/")
def read_root():
    return FileResponse("webcam.html")

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    global lastSaveTime
    image_bytes = await file.read()
    img = Image.open(BytesIO(image_bytes))
    image_tensor = preprocess_image(img, val_transform)
    emotion, confidence = predict(image_tensor, model)
    
    if (emotion in emotions or confidence >= confidenceThreshold):
        now = time.time()
        async with saveLock:
            if now - lastSaveTime >= cooldown:
                lastSaveTime = now
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                public_id = f"{emotion}_{confidence:.2f}_{timestamp}"
                url = await asyncio.to_thread(upload_image, img, public_id)
                if url:
                    print(f"Saved: {url}")


    return {
        "Emotion":emotion,
        "confidence":round(confidence,4)
    }