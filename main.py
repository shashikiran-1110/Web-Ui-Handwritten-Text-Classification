import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

# Set the title of the web app
st.title('Handwritten Text Classification')

# Load MNIST dataset
mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')
X /= 255.0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Build a Neural Network with PyTorch
device = 'cpu'
mnist_dim = X.shape[1]
hidden_dim = int(mnist_dim / 8)
output_dim = len(np.unique(mnist.target))

# Define the architecture of the neural network
class ClassifierModule(nn.Module):
    def __init__(
            self,
            input_dim=mnist_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=0.5,
    ):
        super(ClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Define the hidden layer and the output layer
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        # Forward pass of the neural network
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X

# Train the neural network
torch.manual_seed(0)
net = NeuralNetClassifier(
    ClassifierModule,
    max_epochs=20,
    lr=0.1,
    device=device,
)
net.fit(X_train, y_train)

# Test the model and calculate accuracy
y_pred = net.predict(X_test)
st.subheader('Test Accuracy (PyTorch model)')
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Test accuracy: {accuracy:.2%}')

# Build a Convolutional Neural Network (CNN)
XCnn = X.reshape(-1, 1, 28, 28)
XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=42)

# Define the architecture of the CNN
class Cnn(nn.Module):
    def __init__(self, dropout=0.5):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(1600, 100) # 1600 = number channels * width * height
        self.fc2 = nn.Linear(100, 10)
        self.fc1_drop = nn.Dropout(p=dropout)

    def forward(self, x):
        # Forward pass of the CNN
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # flatten over channel, height and width = 1600
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        x = torch.relu(self.fc1_drop(self.fc1(x)))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

# Train the CNN
torch.manual_seed(0)
cnn = NeuralNetClassifier(
    Cnn,
    max_epochs=10,
    lr=0.002,
    optimizer=torch.optim.Adam,
    device=device,
)
cnn.fit(XCnn_train, y_train)

# Test the model and calculate accuracy
y_pred_cnn = cnn.predict(XCnn_test)
st.subheader('Test Accuracy (CNN)')
accuracy_cnn = accuracy_score(y_test, y_pred_cnn)
st.write(f'Test accuracy: {accuracy_cnn:.2%}')

# Define a function to generate a random digit
def generate_random_digit():
    random_digit = np.random.randint(0, 10)  # Generate a random digit (0-9)
    st.write(f"Randomly generated digit: {random_digit}")
    return random_digit

# Generate a random digit
random_digit = generate_random_digit()

# Preprocess the image data with Pillow
img_data = X_test[random_digit]
img_data = img_data.reshape(-1, 1, 28, 28)

# Predict the digit using your CNN model
pred = cnn.predict(img_data)

# Display the predicted digit
st.subheader('Predicted digit:')
st.write(f'Predicted digit: {pred[0]}')
