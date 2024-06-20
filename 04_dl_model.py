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
    

def shuffle_segments_inside(df, segment_size):
    """
    Shuffle segments within each class in a DataFrame while keeping the order within segments.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing at least 'Activity' and 'Datetime_linacc' columns.
    - segment_size (int): The number of rows in each segment to shuffle.

    Returns:
    - pd.DataFrame: DataFrame with shuffled segments.
    """
    # Check if necessary columns are present
    if 'Activity' not in df.columns or 'Datetime_linacc' not in df.columns:
        raise ValueError("DataFrame must include 'Activity' and 'Datetime_linacc' columns")

    # Add a segment ID based on class label
    df['segment_id'] = df.groupby('Activity').cumcount() // segment_size

    # Shuffle segments within each class
    shuffled_df = pd.DataFrame()
    for label, group in df.groupby('Activity'):
        # Shuffle segments
        shuffled_segments = group.groupby('segment_id').apply(lambda x: x.sample(frac=1).reset_index(drop=True))
        shuffled_segments.index = shuffled_segments.index.droplevel(0)  # Drop the segment level index
        shuffled_df = pd.concat([shuffled_df, shuffled_segments])
    
    # Drop the temporary 'segment_id' column
    shuffled_df.drop('segment_id', axis=1, inplace=True)

    # Return the shuffled DataFrame
    return shuffled_df


def shuffle_segments_global(df, segment_size):
    """
    Shuffle segments within a DataFrame such that each segment contains only one type of activity,
    but segments are shuffled globally across the dataset.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing at least 'Activity' and 'Datetime_linacc' columns.
    - segment_size (int): The number of rows in each segment to shuffle.

    Returns:
    - pd.DataFrame: DataFrame with globally shuffled segments.
    """
    # Check if necessary columns are present
    if 'Activity' not in df.columns or 'Datetime_linacc' not in df.columns:
        raise ValueError("DataFrame must include 'Activity' and 'Datetime_linacc' columns")

    # Add a segment ID based on class label
    df['segment_id'] = df.groupby('Activity').cumcount() // segment_size

    # Collect all segments
    segments = []
    for label, group in df.groupby('Activity'):
        # Collect segments
        for _, segment_group in group.groupby('segment_id'):
            segments.append(segment_group)

    # Shuffle all collected segments
    np.random.shuffle(segments)

    # Concatenate all segments back into one DataFrame
    shuffled_df = pd.concat(segments).reset_index(drop=True)

    # Drop the temporary 'segment_id' column
    shuffled_df.drop('segment_id', axis=1, inplace=True)

    # Return the shuffled DataFrame
    return shuffled_df


# Preprocess the data
def preprocess_data(df, target_column, window_size):
    le = LabelEncoder()
    df = df.copy()
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
    print('Reading dataset...')
    df = pd.read_csv('data_agg/point_two_sec_agg_clean.csv', index_col=0)
    print('Dataset loaded with dimensions:', df.shape)

    # Resetting index
    print('Resetting indices...')
    df.reset_index(drop=False, inplace=True)  

    # Display DataFrame structure
    print("Columns in dataset:", df.columns)
    print("Index type in dataset:", type(df.index))

    # # Shuffle segments inside classes
    # print("Shuffling training dataset segments...")
    # train_df = shuffle_segments_inside(train_df, 30)
    # print("Shuffling testing dataset segments...")
    # test_df = shuffle_segments_inside(test_df, 30)


    # Shuffle segments globally classes
    segment_size = 100
    print("Shuffling dataset segments...")
    df = shuffle_segments_global(df, segment_size)

    # Split the dataset for training and testing
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # Preprocess the data
    window_size = 10
    print("Preprocessing the data...")
    X_train, y_train, classes = preprocess_data(train_df, 'Activity', window_size)
    X_test, y_test, _ = preprocess_data(test_df, 'Activity', window_size)
    print("Data preprocessed with window size:", window_size)

    # Standardize the features
    print("Standardizing the features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    print("Features standardized")

    # Convert to PyTorch tensors
    print("Converting data to tensors...")
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    print("Data conversion complete")

    # Create DataLoader
    batch_size = 64
    print("Creating data loaders...")
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Loaders created")

    # Define the model, criterion, and optimizer
    print("Initializing the model...")
    input_size = X_train.shape[2]
    hidden_size = 64
    num_layers = 3
    num_classes = len(classes)
    num_epochs = 30
    learning_rate = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    print("Training complete")

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, test_loader, classes)
    print("Evaluation done. Process completed.")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
