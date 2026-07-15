from model import EmotionCNN
import torch
from torchvision import transforms
from  PIL import Image
import cv2
import numpy as np

model = EmotionCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load("emotion_cnn.pt", map_location=device))
model.eval()
model = model.to(device)

val_transform = transforms.Compose([
    transforms.ToTensor(),
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_and_crop_face(pil_image):
    """Returns a cropped PIL Image of the largest detected face, or None."""
    img_np = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None

    # Pick the largest detected face (closest to camera / main subject)
    largest_face = max(faces, key=lambda box: box[2] * box[3])
    x, y, w, h = largest_face
    return pil_image.crop((x, y, x + w, y + h))

def preprocess_image(img, transform):
    face = detect_and_crop_face(img)
    if face is not None:
        img = face
    img = img.convert('L')
    img = img.resize((48, 48))
    image_tensor = transform(img)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

emotions = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

def predict(image_tensor, model):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        res = torch.argmax(probs, dim=1)
        label = res.item()
    confidence = probs[0, res.item()].item()
    return emotions[label], confidence

if __name__ == "__main__":
    image_path = "PrivateTest_33585800.jpg"
    image_tensor = preprocess_image(Image.open(image_path), val_transform)
    emote, confidence = predict(image_tensor, model)
    print(emote, confidence)
        