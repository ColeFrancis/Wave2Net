import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from SignalDataset import SignalDataset

from model import Model

################################################################################
### Load in the data
################################################################################


dataset = SignalDataset("./Dataset", split="test", task="waveform")
loader = torch.utils.data.DataLoader(dataset, batch_size=1)

checkpoint = torch.load("trained_model.pth")

class_map = checkpoint["class_map"]
num_classes = len(class_map)

model = Model(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

idx_to_class = {v: k for k, v in class_map.items()}
correct = 0
total = 0

with torch.no_grad():
    for signal, label in loader:
        outputs = model(signal)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)

        predicted_idx = torch.argmax(outputs, dim=1).item()

        # Convert index → actual label
        predicted_value = idx_to_class[predicted_idx]
        true_value = idx_to_class[label.item()]

        print(f"Predicted: {predicted_value}")
        print(f"Actual:    {true_value}")

print(f"Accuracy: {correct/total:.4f}")
