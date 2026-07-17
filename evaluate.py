import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from dataset import getDatasets
from model import EmotionCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, val_dataset = getDatasets("fer2013.csv")
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = EmotionCNN(num_classes=7)
model.load_state_dict(torch.load("emotion_cnn.pt", map_location=device))
model.eval()
model = model.to(device)

correct = 0
total = 0
class_correct = defaultdict(int)
class_total = defaultdict(int)

emotions = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
            4: "Sad", 5: "Surprise", 6: "Neutral"}

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        for pred, true in zip(predictions, labels):
            class_total[true.item()] += 1
            if pred == true:
                class_correct[true.item()] += 1

print(f"Overall accuracy: {correct/total:.4f}\n")
for i in range(7):
    acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"{emotions[i]:>10}: {acc:.3f}  ({class_total[i]} samples)")