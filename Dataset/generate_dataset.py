import numpy as np
import random
from scipy import signal

SAMPLE_RATE = 44100
NUM_SAMPLES = 1024

FREQS = [200, 500, 1000, 2000, 5000, 10000]

MIN_AMPLITUDE = -1.0
MAX_AMPLITUDE = 1.0

WAVE_TYPE = ['sine', 'square', 'triange', 'saw', 'noise']

t = np.arrange(0, 1024)

def generate_random_sample(type):
    freq = FREQS[np.random.randint(0, len(FREQS)-1)]

    amplitude = np.random.uniform(MIN_AMPLITUDE, MAX_AMPLITUDE)

    phase = np.random.uniform(0, 2*np.pi)

    match type:
        case 'sine':
            x = amplitude * np.sin(2*np.pi * freq * t + phase)

        case 'square':
            x = amplitude * signal.square(2*np.pi * freq * t + phase)

        case 'triange':
            x = amplitude * signal.sawtooth(2*np.pi * freq * t + phase, width=0.5)

        case 'saw':
            x = amplitude * signal.sawtooth(2*np.pi * freq * t + phase)

        case 'noise':
            x = amplitude * [random.random() for _ in range(len(t))]

        case _:
            x = np.zeros_like(t)

    # add noise

    return x
