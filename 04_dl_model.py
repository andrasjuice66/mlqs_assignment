import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Preprocess the data
def preprocess_data(df, target_column, window_size):
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
    
    # Convert datetime columns to numerical values
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                df[col] = df[col].astype('int64') // 10**9  # convert to seconds
            except ValueError:
                pass

    # Check for missing values and fill or drop them
    df = df.fillna(df.mean())
    
    sequences = []
    labels = []
    for i in range(len(df) - window_size):
        sequences.append(df.iloc[i:i + window_size].drop(columns=[target_column]).values)
        labels.append(df.iloc[i + window_size][target_column])
    
    X = np.array(sequences)
    y = np.array(labels)
    
    return X, y, le.classes_

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
def evaluate_model(model, test_loader, classes):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Main function
def main():
    # Read datasets
    print('Read training dataset')
    train_df = pd.read_csv('data_agg/feature_engineered_train.csv', index_col=0)
    print('Read testing dataset')
    test_df = pd.read_csv('data_agg/feature_engineered_test.csv', index_col=0)

    # Preprocess the data
    window_size = 5
    X_train, y_train, classes = preprocess_data(train_df, 'Activity', window_size)
    X_test, y_test, _ = preprocess_data(test_df, 'Activity', window_size)
    print("Data preprocessed")
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader
    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Loaders created")
    # Define the model, criterion, and optimizer
    input_size = X_train.shape[2]
    hidden_size = 64
    num_layers = 2
    num_classes = len(classes)
    num_epochs = 20
    learning_rate = 0.001

    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    print("Trained")
    # Evaluate the model
    evaluate_model(model, test_loader, classes)
    print("Done")
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
