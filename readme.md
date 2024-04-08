# Traffic Light Recognition Using CNN for Autonomous driving

## Project Overview

This project develops a traffic light recognition system specifically tailored for autonomous vehicles. Using a Convolutional Neural Network (CNN), this system classifies images of traffic lights into four categories: Red, Yellow, Green. The goal is to provide a reliable assistant to autonomous vehicle navigation systems by accurately interpreting traffic light signals.

## Features

- **Image Classification**: Classify traffic light images into Red, Yellow, Green.
- **CNN Model**: Leverages a convolutional neural network for high accuracy.
- **Data Augmentation**: Implements data augmentation to enhance model robustness.

## Installation

Follow these steps to set up and run the project:

1. pip install -r requirements.txt
-**Python packages**:
- TensorFlow
- Keras
- NumPy
- Pillow

2. Run the **py traffic_light_recognition_cnn.py** command in cmd to train the dataset and generate the model file(.h5)

3. To identify the traffic light, Run **py recognition.py** command in cmd and update the image path in the python file. 
