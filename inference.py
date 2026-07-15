from model import EmotionCNN
import torch
from torchvision import transforms
from  PIL import Image

model = EmotionCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load("emotion_cnn.pt", map_location=device))
model.eval()
model = model.to(device)

val_transform = transforms.Compose([
    transforms.ToTensor(),
])

def preprocess_image(img, transform):
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
        