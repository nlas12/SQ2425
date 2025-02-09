# SQ2425

## Description
Implementation of oarriaga/paz and face classification in the context of the Software Quality course at the University of Cologne.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)
- [Contact](#contact)

## Installation
The oarriaga/paz repository **cannot** be installed with the package manager as the version is deprecated. Please clone or download the repository directly.

```sh
git clone https://github.com/nlas12/SQ2425.git
cd SQ2425
pip install -r requirements
cd ..

git clone https://github.com/oarriaga/paz.git
cd paz
cp -r paz/paz ../SQ2425/ #copy paz/paz into SQ2425
```

## Usage

Example data data from https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition.
Create a data folder inside the project directory and place the dataset there.

Use the dataset above in conjunction with the notebooks to create confusion matrices for both deepface and paz.

### Scripts

Plug in webcam, navigate to repository and run:
```sh
python ./detection_deepface.py
```
or
```sh
python ./detection_paz.py
```
depending on which model you want to use.

## Features

- Semi-Real-time Emotion Recognition: Captures frames from a webcam and detects emotions.
- DeepFace Model: Uses the DeepFace library for emotion classification.
- PAZ Model: Implements the MiniXceptionFER model for emotion recognition.

## License

This project falls under the MIT license.

## Contact

For inquiries, feel free to reach out via GitHub.