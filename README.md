# Wave2Net
Train a custom neural network to classify a signals waveform and frequency.

This repository was created for the Wav2Net workshop at Utah State University's (USU) 2026 Student Professional Awareness Conference (SPAC) put on by USU's IEEE Student Branch.

# Files and Directories

### Dataset/

Contains the data to train the model on. 

The signal samples are .txt files with the signal magnitude at each succesive time step on the next line. The labels are in json files.

### SignalDataset.py

This python file builds a helper class around the dataset for easy usage.

### util.py

This file contains helper functions.

### examples/

Contains an example model with example training and prediction scripts.
