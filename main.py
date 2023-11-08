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

# Set the title of the web app
st.title('Handwritten Text Classification')

# Load MNIST dataset
st.subheader('1. Load and Preprocess Data')
mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')
X /= 255.0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Show a selection of training images and their labels
st.subheader('2. Explore the Data')
st.write("Example training images and labels:")

# Display 16 training images and their labels in a 4x4 grid
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(y_train[i])
    plt.axis('off')
st.pyplot(plt)

# Build a Neural Network with PyTorch
st.subheader('3. Build a Neural Network')
device = 'cuda' if torch.cuda.is available() else 'cpu'
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
st.subheader('4. Train the Neural Network (PyTorch)')
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
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Test accuracy (PyTorch model): {accuracy:.2%}')
# Visualize some misclassified images
error_mask = y_pred != y_test
st.subheader('5. Visualize Misclassified Images (PyTorch model)')

# Display up to 16 misclassified images and their predicted labels
plt.figure(figsize=(10, 10))
for i in range(min(16, sum(error_mask))):
    plt.subplot(4, 4, i + 1)
    plt.imshow(X_test[error_mask][i].reshape(28, 28), cmap='gray')
    plt.title(y_pred[error_mask][i])
    plt.axis('off')
st.pyplot(plt)
