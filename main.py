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
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
from torchvision import transforms


# Set the title of the web app
st.title('Handwritten Text Classification')

# Load MNIST dataset
st.subheader('1. Load and Preprocess Data')
mnist = fetch_openml('mnist_784', as_frame=False, cache=False, parser='auto')
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

# Build a Convolutional Neural Network (CNN)
st.subheader('6. Build a Convolutional Neural Network (CNN)')
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
st.subheader('7. Train the Convolutional Neural Network (CNN)')
torch.manual_seed(0)
cnn = NeuralNetClassifier(
    Cnn,
    max_epochs=10,
    lr=0.002,
    optimizer=torch.optim.Adam,
    device=device,
)
cnn.fit(XCnn_train, y_train)

# Test the CNN model and calculate accuracy
y_pred_cnn = cnn.predict(XCnn_test)
accuracy_cnn = accuracy_score(y_test, y_pred_cnn)
st.write(f'Test accuracy (CNN): {accuracy_cnn:.2%}')

# Visualize some misclassified images by the CNN model
st.subheader('8. Visualize Misclassified Images (CNN)')

# Display up to 16 misclassified images and their predicted labels by the CNN
plt.figure(figsize=(10, 10))
for i in range(min(16, sum(error_mask))):
    plt.subplot(4, 4, i + 1)
    plt.imshow(X_test[error_mask][i].reshape(28, 28), cmap='gray')
    plt.title(y_pred_cnn[error_mask][i])
    plt.axis('off')
st.pyplot(plt)

# Define a function to generate a random digit image
def generate_random_digit():
    random_digit = np.random.randint(0, 10)  # Generate a random digit (0-9)
    img = Image.new("L", (28, 28), color=0)  # Create a blank 28x28 image
    img_data = img.load()
    st.write(f"Randomly generated digit: {random_digit}")

    # Create the digit image
    for i in range(28):
        for j in range(28):
            img_data[j, i] = 255  # Set pixel to white

    return img, random_digit

# Set the title of the web app
st.title('Handwritten Text Classification')

# Display a randomly generated digit
digit_image, random_digit = generate_random_digit()
st.image(digit_image, caption=f'Random Digit: {random_digit}', use_column_width=True)

# Load your trained CNN model (cnn) here

# Preprocess the image data with Pillow
img_data = digit_image.resize((28, 28))
img_data = np.array(img_data).astype('float32') / 255.0
img_data = img_data.reshape(-1, 1, 28, 28)

# Predict the digit using your CNN model
pred = cnn.predict(img_data)

# Display the predicted digit
st.title('Predicted digit:')
st.write(f'Predicted digit: {pred[0]}')




