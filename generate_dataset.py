import json
import random
import shutil
from pathlib import Path
import numpy as np
from scipy import signal

config_file = "./dataset_config.json"

with open(config_file, "r") as f:
    configs = json.load(f)

base_dir = Path(configs["datasetConfigs"]["datasetName"])

if base_dir.exists():
    shutil.rmtree(base_dir)

def generate_random_sample(times, wave_type, signalConfigs):
    freq = signalConfigs["frequencies"][np.random.randint(0, len(signalConfigs["frequencies"])-1)]

    amplitude = np.random.uniform(signalConfigs["minAmplitude"], signalConfigs["maxAmplitude"])

    phase = np.random.uniform(0, 2*np.pi)

    match wave_type:
        case 'sine':
            x = amplitude * np.sin(2*np.pi * freq * times + phase)

        case 'square':
            x = amplitude * signal.square(2*np.pi * freq * times + phase)

        case 'triange':
            x = amplitude * signal.sawtooth(2*np.pi * freq * times + phase, width=0.5)

        case 'saw':
            x = amplitude * signal.sawtooth(2*np.pi * freq * times + phase)

        case 'noise':
            x = amplitude * np.array([random.random() for _ in range(len(times))])

        case _:
            x = np.zeros_like(times)

    # add noise
    noise = np.random.normal(0, signalConfigs["noiseScale"], size= x.shape)

    return x + noise, freq

t = np.arange(configs["signalConfigs"]["numSamples"]) / configs["signalConfigs"]["sampleRate"]

signal_types = configs["signalConfigs"]["signalTypes"]


for split in ["train", "test"]:

    num_samples = (
        configs["datasetConfigs"]["numTrain"]
        if split == "train"
        else configs["datasetConfigs"]["numTest"]
    )
    file_name_width = len(str(num_samples - 1))

    samples_dir = base_dir / split / "samples"
    labels_dir = base_dir / split / "labels"

    samples_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(signal_types)
    samples_per_class = num_samples // num_classes

    # Build balanced list of wave types
    wave_list = []
    for wave in signal_types:
        wave_list.extend([wave] * samples_per_class)

    # If num_samples not divisible, fill remainder
    remainder = num_samples - len(wave_list)
    for i in range(remainder):
        wave_list.append(signal_types[i % num_classes])

    # Shuffle so dataset isn't ordered by class
    random.shuffle(wave_list)

    for i, wave_type in enumerate(wave_list):

        sig, freq = generate_random_sample(
            t,
            wave_type,
            configs["signalConfigs"]
        )

        file_index = f"{i:0{file_name_width}d}"
        
        # Save signal
        sample_path = samples_dir / f"{file_index}.txt"
        with open(sample_path, "w") as f:
            for value in sig:
                f.write(f"{value}\n")

        # Save label
        label_data = {
            "wave_type": wave_type,
            "frequency": float(freq)
        }

        label_path = labels_dir / f"{file_index}.json"
        with open(label_path, "w") as f:
            json.dump(label_data, f, indent=4)

print("Dataset generation complete.")