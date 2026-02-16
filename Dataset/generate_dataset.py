import json
import random
import shutil
from pathlib import Path
import numpy as np
from scipy import signal

config_file = "./generate_dataset_config.json"

with open(config_file, "r") as f:
    configs = json.load(f)

base_dir = Path(configs["datasetConfigs"]["datasetName"])

if base_dir.exists():
    shutil.rmtree(base_dir)

for split in ["train", "test"]:
    (base_dir / split / "samples").mkdir(parents=True, exist_ok=True)
    (base_dir / split / "labels").mkdir(parents=True, exist_ok=True)

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

    return x + noise


import matplotlib.pyplot as plt


t = np.arange(configs["signalConfigs"]["numSamples"]) / configs["signalConfigs"]["sampleRate"]

sig = generate_random_sample(t, configs["signalConfigs"]["signalTypes"][0], configs["signalConfigs"])

plt.plot(t, sig)
plt.show()