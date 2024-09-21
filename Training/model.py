import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
# LSTM
# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load your data (assuming numerical features and emotion labels)
data = pd.read_csv('cleaned_file2.csv')

# Assuming the CSV has a 'emotion_label' column with the labels and the rest are numerical features
X = data.drop(columns=['emotion_label']).values  # Features
y = data['emotion_label'].values  # Categorical Labels (e.g., 'Anger', 'Joy', etc.)

# Step 1.1: Convert categorical labels to integers using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Converts labels like 'Anger' to integers

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 3: Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Reshape the data for LSTM input (samples, time_steps, features)
# Here, we treat each sample as a sequence with one time step (the entire feature vector is one "time step")
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Step 5: Define a custom dataset class
class AudioFeaturesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)  # Ensure features are float
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are long (int64)
        return feature, label

# Step 6: Create data loaders
train_dataset = AudioFeaturesDataset(X_train_scaled, y_train)
test_dataset = AudioFeaturesDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Step 7: Define the LSTM model
class LSTMAudioClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMAudioClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)  # Initial hidden state
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)  # Initial cell state

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]  # Take the output from the last time step
        out = self.fc(out)  # Fully connected layer (logits)
        return out

# Step 8: Initialize the model
input_size = X_train_scaled.shape[2]  # Number of features (input dimension to LSTM)
hidden_size = 128  # Number of hidden units in LSTM
num_layers = 2  # Number of LSTM layers
num_classes = len(set(y_encoded))  # Number of unique labels (emotions)

model = LSTMAudioClassifier(input_size, hidden_size, num_layers, num_classes)
model.to(device)

# Step 9: Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross entropy for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 10: Training loop
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Step 11: Evaluation loop
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels

# Step 12: Train the model
num_epochs = 10  # Adjust number of epochs as necessary
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss}")

# Step 13: Evaluate on the test set
y_pred, y_true = evaluate(model, test_loader)

# Step 14: Print classification results
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Step 15: Save the model
model_save_path = 'KCG_LSTM_model2'
torch.save(model.state_dict(), model_save_path)
