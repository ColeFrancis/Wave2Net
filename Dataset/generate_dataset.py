import numpy as np
import random
from scipy import signal

SAMPLE_RATE = 44100
NUM_SAMPLES = 256# 1024
# A4 E5 A5 E6 A6 E7
FREQS = [440.0, 659.3, 880.0, 1318.5, 1760.0, 2637.0]#[200, 500, 1000, 2000, 5000, 10000]

MIN_AMPLITUDE = -1.0
MAX_AMPLITUDE = 1.0

WAVE_TYPE = ['sine', 'square', 'triange', 'saw', 'noise']

t = np.arange(NUM_SAMPLES) / SAMPLE_RATE

def generate_random_sample(wave_type):
    freq = FREQS[np.random.randint(0, len(FREQS)-1)]

    amplitude = np.random.uniform(MIN_AMPLITUDE, MAX_AMPLITUDE)

    phase = np.random.uniform(0, 2*np.pi)

    match wave_type:
        case 'sine':
            x = amplitude * np.sin(2*np.pi * freq * t + phase)

        case 'square':
            x = amplitude * signal.square(2*np.pi * freq * t + phase)

        case 'triange':
            x = amplitude * signal.sawtooth(2*np.pi * freq * t + phase, width=0.5)

        case 'saw':
            x = amplitude * signal.sawtooth(2*np.pi * freq * t + phase)

        case 'noise':
            x = amplitude * np.array([random.random() for _ in range(len(t))])

        case _:
            x = np.zeros_like(t)

    # add noise
    noise = np.random.normal(0, 0.01, size= x.shape)

    return x + noise


import matplotlib.pyplot as plt

sig = generate_random_sample(WAVE_TYPE[0])

plt.plot(t, sig)
plt.show()