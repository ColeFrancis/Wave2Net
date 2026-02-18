import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from SignalDataset import SignalDataset

from model import Model

################################################################################
### Load in the data
################################################################################

TASK = "waveform"
DATASET_ROOT = './Dataset'
BATCH_SIZE = 64

train_dataset = SignalDataset(DATASET_ROOT, split="train", task=TASK)
test_dataset = SignalDataset(DATASET_ROOT, split="test", task=TASK)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

num_classes = len(train_dataset.class_map)
input_size = len(train_dataset[0][0])


################################################################################
### Training
################################################################################

LEARNING_RATE = 1e-3
EPOCHS = 20

model = Model(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, label in train_loader:
        data = data.unsqueeze(1)
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)

    print(
        f"Epoch {epoch+1}, "
        f"Loss: {total_loss/len(train_loader):.4f}, "
        f"Accuracy: {correct/total:.4f}"
    )

torch.save(model.state_dict(), "trained_model.pth")
print("Model Saved!")