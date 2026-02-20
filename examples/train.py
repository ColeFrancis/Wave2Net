import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# This is only necessary to import SignalDataset because it is not in the examples directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from SignalDataset import SignalDataset

from model_cnn import ModelCNN

################################################################################
### Load in the data
################################################################################

TASK = "waveform"
#TASK = "frequency"
DATASET_ROOT = '../Dataset'
BATCH_SIZE = 64

# Creates the dataset helper class on the training data to predict the TASK
train_dataset = SignalDataset(DATASET_ROOT, split="train", task=TASK)

# loader will grab batch_size chunks of data at a time
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

input_size = len(train_dataset[0][0])
num_classes = len(train_dataset.class_map)

################################################################################
### Training
################################################################################

LEARNING_RATE = 1e-3
EPOCHS = 20

model = ModelCNN(input_size, num_classes)

# criterion evaluates how wrong the model is, optimizer updates the weights as we train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# each epoch loops over the entire dataset
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # loops over each chunk grabbed by the loader
    for data, label in train_loader:
        # Prepare optimizer
        optimizer.zero_grad()

        # Run the chunk of data thrugh the model and determine how wrong it is
        outputs = model(data)
        loss = criterion(outputs, label)

        # Update the weights
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Determine which predictions were correct and how many there were
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)

    print(
        f"Epoch {epoch+1}, "
        f"Loss: {total_loss/len(train_loader):.4f}, "
        f"Accuracy: {correct/total:.4f}"
    )

# Save the weights, as well as the mapping to know which output index corresponds to which waveform/frequency
torch.save({
    "model_state_dict": model.state_dict(),
    "idx_to_class": train_dataset.idx_to_class
}, "trained_model.pth")
print("Model Saved!")
