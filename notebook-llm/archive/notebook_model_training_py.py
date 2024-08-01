import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error

from dotenv_vault import load_dotenv
load_dotenv()

# Load the CSV file
csv_file = os.getenv('TOKENIZE_NOTES')
df = pd.read_csv(csv_file)

# Drop the 'filename' column as it is not needed for training
df = df.drop(columns=['filename'])

# Split data into features and labels
X = df.drop(columns=['word_count', 'content_length'])
y = df['word_count']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Normalize the target (if needed)
y_train_mean = y_train.mean()
y_train_std = y_train.std()
y_train_scaled = (y_train - y_train_mean) / y_train_std
y_test_scaled = (y_test - y_train_mean) / y_train_std

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled.values, dtype=torch.float32)

class NotesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create datasets
train_dataset = NotesDataset(X_train_tensor, y_train_tensor)
test_dataset = NotesDataset(X_test_tensor, y_test_tensor)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class EnhancedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EnhancedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)  # For regression, output a single value

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
# Model parameters
input_dim = X_train_tensor.shape[1]
hidden_dim = 750

# Initialize the model
model = EnhancedNN(input_dim, hidden_dim)

criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training function with more epochs
def train_model(model, train_loader, criterion, optimizer, num_epochs=300):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

# Train the model
train_model(model, train_loader, criterion, optimizer)


# Inverse transform the predictions and target values (if normalized)
def inverse_transform(y_scaled):
    return y_scaled * y_train_std + y_train_mean

def evaluate_model(model, test_loader):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            all_predictions.extend(outputs.squeeze().tolist())
            all_labels.extend(labels.tolist())

    # Inverse transform the predictions and target values (if normalized)
    predictions = inverse_transform(np.array(all_predictions))
    actuals = inverse_transform(np.array(all_labels))
    
    # Print a few predictions and actual values for inspection
    for i in range(10):  # Print first 10 for inspection
        print(f"Predicted: {predictions[i]}, Actual: {actuals[i]}")
    
    avg_loss = total_loss / len(test_loader)
    print(f'Average Loss: {avg_loss}')

# Evaluate the model
evaluate_model(model, test_loader)

def evaluate_model_metrics(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            all_predictions.extend(outputs.squeeze().tolist())
            all_labels.extend(labels.tolist())

    # Inverse transform the predictions and target values (if normalized)
    predictions = inverse_transform(np.array(all_predictions))
    actuals = inverse_transform(np.array(all_labels))
    
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(actuals, predictions)
    
    # Print a few predictions and actual values for inspection
    for i in range(10):  # Print first 10 for inspection
        print(f"Predicted: {predictions[i]}, Actual: {actuals[i]}")
    
    print(f'Mean Absolute Error: {mae}')

# Evaluate the model with metrics
evaluate_model_metrics(model, test_loader)


# save output of the model
from dotenv_vault import load_dotenv
load_dotenv()

model_pt_file = os.getenv('MODEL_PT_FILE')

torch.save(model, model_pt_file)