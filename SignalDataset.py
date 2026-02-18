import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np


class SignalDataset(Dataset):
    def __init__(self, root_dir, split="train", task="waveform"):
        """
        task:
            "waveform"  -> classify wave type
            "frequency" -> classify frequency
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.task = task

        self.samples_dir = self.root_dir / split / "samples"
        self.labels_dir = self.root_dir / split / "labels"

        self.sample_files = sorted(self.samples_dir.glob("*.txt"))

        # Build class mapping depending on task
        values = set()

        for label_file in self.labels_dir.glob("*.json"):
            with open(label_file) as f:
                data = json.load(f)

                if task == "waveform":
                    values.add(data["wave_type"])
                elif task == "frequency":
                    values.add(float(data["frequency"]))

        self.class_map = {
            val: idx for idx, val in enumerate(sorted(values))
        }

        self.idx_to_class = {
            v: k for k, v in class_map.items()
        }

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample_path = self.sample_files[idx]
        label_path = self.labels_dir / (sample_path.stem + ".json")

        # Load signal
        signal = np.loadtxt(sample_path).astype(np.float32)
        signal = torch.tensor(signal)

        #Add dimension for channel width
        signal = signal.unsqueeze(0)

        # Load label
        with open(label_path) as f:
            data = json.load(f)

        if self.task == "waveform":
            label_value = data["wave_type"]
        else:
            label_value = float(data["frequency"])

        label = torch.tensor(
            self.class_map[label_value],
            dtype=torch.long
        )

        return signal, label
