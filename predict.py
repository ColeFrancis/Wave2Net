import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from SignalDataset import SignalDataset
from util import plot_random_signal

from model import Model

################################################################################
### Load in the data
################################################################################

#TASK = "waveform"
TASK = "frequency"

dataset = SignalDataset("./Dataset", split="test", task=TASK)
loader = torch.utils.data.DataLoader(dataset, batch_size=1)

################################################################################
### Load the Model
################################################################################

checkpoint = torch.load("trained_model.pth")

idx_to_class = checkpoint["idx_to_class"]
num_classes = len(idx_to_class)

model = Model(num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

################################################################################
### Evaluate the model
################################################################################

correct = 0
total = 0

with torch.no_grad():
    for signal, label in loader:
        outputs = model(signal)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)

print(f"Accuracy: {correct/total:.4f}")

plot_random_signal(dataset, model, idx_to_class)
