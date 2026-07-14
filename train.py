import torch
from torch.utils.data import DataLoader
from dataset import getDatasets
from model import EmotionCNN

train_dataset, val_dataset = getDatasets("fer2013.csv")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

model = EmotionCNN(num_classes=7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
lossf = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

epochs = 20
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = lossf(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    scheduler.step()

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = lossf(logits, labels)
            val_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct/total
    print(f"Epoch {epoch+1}/{epochs} | train loss {avg_train_loss:.4f} | "
          f"val loss {avg_val_loss:.4f} | val acc {val_accuracy:.4f}")
    
torch.save(model.state_dict(), "emotion_cnn.pt")