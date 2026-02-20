import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# This is only necessary to import SignalDataset because it is not in the examples directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from SignalDataset import SignalDataset
from util import plot_random_signal

from model_cnn import ModelCNN

################################################################################
### Load in the data
################################################################################

TASK = "waveform"
#TASK = "frequency"
DATASET_ROOT = '../Dataset'

# Creates the dataset helper class on the training data to predict the TASK
dataset = SignalDataset(DATASET_ROOT, split="test", task=TASK)
# For predicting, we only want a batch_size of 1 to grab one sample at a time
loader = torch.utils.data.DataLoader(dataset, batch_size=1)

################################################################################
### Load the Model
################################################################################

input_size = len(dataset[0][0])

# Load the saved weights and mapping from output index to waveform/frequency
checkpoint = torch.load("trained_model.pth")

# Determine how many output waveforms/frequencies there are
idx_to_class = checkpoint["idx_to_class"]
num_classes = len(idx_to_class)

# Load the model and set it to "eval" mode (predicting, not training)
model = ModelCNN(input_size, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

################################################################################
### Evaluate the model
################################################################################

correct = 0
total = 0

# torch.no_grad tells the program we aren't training the model, only predicting
with torch.no_grad():
    # loop over each sample grabbed by the loader
    for signal, label in loader:
        # Run the data through the model
        outputs = model(signal)

        # Count how many predictions were correct
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)

print(f"Accuracy: {correct/total:.4f}")

plot_random_signal(dataset, model, idx_to_class)
