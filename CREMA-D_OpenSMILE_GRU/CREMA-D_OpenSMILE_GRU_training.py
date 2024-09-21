import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load your data (assuming numerical features and emotion labels)
data = pd.read_csv('yio.csv')

# Assuming the CSV has a 'emotion_label' column with the labels and the rest are numerical features
X = data.drop(columns=['emotion_label']).values  # Features
y = data['emotion_label'].values  # Categorical Labels (e.g., 'Anger', 'Joy', etc.)

# Step 1.1: Convert categorical labels to integers using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # This converts labels like 'Anger' to integers

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 3: Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Define a custom dataset class
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

# Step 5: Create data loaders
train_dataset = AudioFeaturesDataset(X_train_scaled, y_train)
test_dataset = AudioFeaturesDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Step 6: Define the GRU model
class GRUNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, 128, batch_first=True)  # GRU layer
        self.fc = nn.Linear(128, num_classes)  # Output layer
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension (batch_size, seq_len=1, input_size)
        out, _ = self.gru(x)  # Forward pass through GRU
        out = self.fc(out[:, -1, :])  # Get the last output (batch_size, num_classes)
        return out

# Step 7: Initialize the model
input_size = X_train_scaled.shape[1]  # Number of features
num_classes = len(set(y_encoded))  # Number of unique labels (emotions)
model = GRUNetwork(input_size, num_classes)
model.to(device)

# Step 8: Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross entropy for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 9: Training loop (No early stopping)
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

# Step 10: Evaluation loop
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), all_preds, all_labels

# Step 11: Training with more epochs
num_epochs = 50  # Increase the number of epochs for more training cycles
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss, y_pred, y_true = evaluate(model, test_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

# Step 12: Evaluate on the test set
_, y_pred, y_true = evaluate(model, test_loader)

# Step 13: Print classification results
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Step 14: Save the final model
model_save_path = 'KCG_GRU_model.pth'
torch.save(model.state_dict(), model_save_path)
