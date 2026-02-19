# Wave2Net
Train a custom network to classify signals b waveform and frequency

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