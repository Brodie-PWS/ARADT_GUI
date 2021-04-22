# About ARADT

ARADT (Audio Replay Attack Detection Toolkit) is an interactive Python Tkinter GUI application which offers functionality for carrying out voice liveness based, machine learning experiments. It also offers the ability to create custom models which can detect Replay attacks utilizing the techniques described in the 'VOID: A fast and light voice liveness detection system' paper.

The system was designed and built as part of a dissertation project.

Supported Features:
- Ability to record, playback and manage audio samples
- Ability to analyse audio samples and display plots related to audio characteristics
- Ability to quickly create and edit custom labels files
- Ability to extract the 'VOID' feature set from audio samples
- Ability to create and save SVM, Random Forest, Multi-layer Perceptron and KNN models which can detect replay attacks
- Ability to automatically determine the best combination of Hyperparameters for a given model and dataset
- Ability to use models to make predictions on audio samples
- Ability to evaluate the performance of models in making predictions
- Ability to store predictions and view them at a later stage

## Resources Used:
Throughout the developement of ARADT, these resources were used:

- [The 'VOID' paper](https://www.usenix.org/conference/usenixsecurity20/presentation/ahmed-muhammad)
- [An Open Source Reproduction of the VOID system](https://github.com/chislab/void-voice-liveness-detection) - ARADT borrows some of its implementation from this repository. The feature_extraction.py script was used as the basis for the feature extraction aspects of ARADT
- [The ASVSpoof 2017 Dataset](https://datashare.ed.ac.uk/handle/10283/3055) - This dataset provided an abundance of audio samples with which to train and evaluate models in ARADT. The dataset has not been provided in this repository and can be downloaded from here. The 'VOID' features have already been extracted from each of the sets and are available in the 'Extracted Features' directory

## Installation:

ARADT requires [Python 3.7](https://www.python.org/downloads/release/python-370/) or above to run.

Clone or download the project to your machine:
```sh
git clone https://github.com/Brodie-PWS/ARADT_GUI.git
```

Navigate to the root directory of project and install the dependancies:
```sh
pip install -r requirements.txt
```
