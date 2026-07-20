from fastapi import FastAPI, UploadFile, File
from inference import model, predict, preprocess_image, val_transform
from io import BytesIO
from PIL import Image
from fastapi.responses import FileResponse

app = FastAPI()

@app.get("/")
def read_root():
    return FileResponse("webcam.html")

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(BytesIO(image_bytes))
    image_tensor = preprocess_image(img, val_transform)
    emotion, confidence = predict(image_tensor, model)

    return {
        "Emotion":emotion,
        "confidence":round(confidence,4)
    }