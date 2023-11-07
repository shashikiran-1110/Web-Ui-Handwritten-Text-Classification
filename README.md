# Handwritten Text Classification with Streamlit

This repository contains code for a Streamlit web application that performs handwritten text classification using the MNIST dataset. It demonstrates two machine learning models: a PyTorch-based Neural Network and a Convolutional Neural Network (CNN) for digit recognition.

## Table of Contents

- [Project Description](#project-description)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Description

The project showcases the following:

1. **Loading and Preprocessing the MNIST dataset:** The code loads the MNIST dataset, preprocesses the data, and splits it into training and testing sets.

2. **Building and Training a Neural Network with PyTorch:** A neural network model is defined using PyTorch, trained on the MNIST training data, and tested on the testing data.

3. **Building and Training a Convolutional Neural Network (CNN) with PyTorch:** A CNN model is defined using PyTorch, trained on the MNIST training data, and tested on the testing data.

4. **Visualizing Misclassified Images:** The code provides visualizations of up to 16 misclassified images from both the neural network and the CNN model.

5. **Drawing a Digit and Prediction:** A canvas is included to draw a digit, which is then fed into the trained CNN model for digit recognition.

## Dependencies

The project relies on the following dependencies:

- Python 3.7+
- Streamlit
- Scikit-learn
- PyTorch
- scikit-neuralnetwork (skorch)
- OpenML
- NumPy
- Matplotlib
- OpenCV (cv2)

## Installation

To get started, follow these steps:

1. **Clone this repository:**

   ```bash
   git clone https://github.com/your-username/handwritten-text-classification.git

## Install the required Python packages:

    ```bash
    pip install -r requirements.txt

## Navigate to the project directory:

    ``` bash
    cd handwritten-text-classification

## Run the Streamlit app:

    ```bash
    streamlit run handwritten_text_classification.py

Interact with the application: The application will open in your web browser, allowing you to explore the MNIST dataset and interact with the classification models.
