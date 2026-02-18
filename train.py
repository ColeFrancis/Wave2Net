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
#TASK = "frequency"
DATASET_ROOT = './Dataset'
BATCH_SIZE = 64

train_dataset = SignalDataset(DATASET_ROOT, split="train", task=TASK)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

num_classes = len(train_dataset.class_map)


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

torch.save({
    "model_state_dict": model.state_dict(),
    "idx_to_class": train_dataset.idx_to_class
}, "trained_model.pth")
print("Model Saved!")
