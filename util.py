import matplotlib.pyplot as plt
import torch
import random

def plot_random_signal(dataset, model, idx_to_class):
    model.eval()
    with torch.no_grad():
        rand_idx = random.randint(0, len(dataset) - 1)

        signal, label = dataset[rand_idx]

        signal = signal.unsqueeze(0)

        outputs = model(signal)
        predicted_idx = torch.argmax(outputs, dim=1).item()
        label_idx = label.item()

        plt.figure()
        plt.plot(signal.squeeze())
        plt.title(f"Real: {idx_to_class[label_idx]} | Predicted: {idx_to_class[predicted_idx]}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()