import os
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Define the path to your audio files and CSV file
audio_folder = r'C:\Users\hanus\Videos\programming\KCG\voices'
csv_file = r'C:\Users\hanus\Videos\programming\KCG\combined_file.csv'  # Combined CSV file with features and labels

# Load CSV file with features and labels
df = pd.read_csv(csv_file)

# Print column names to find the correct column for file paths
print("CSV Columns:", df.columns)

# Replace 'file_path' and 'nervousness' with the actual column names
file_path_column = 'file_path1'  # Update this with the actual column name for file paths
label_column = 'nervousness'     # Update this with the actual column name for labels

# Ensure the columns exist
if file_path_column not in df.columns:
    raise ValueError(f"File path column '{file_path_column}' not found in the CSV file.")
if label_column not in df.columns:
    raise ValueError(f"Label column '{label_column}' not found in the CSV file.")

# Prepare file paths and labels
file_paths = df[file_path_column].tolist()
labels = df[label_column].values

# Check unique labels and ensure they are within the expected range
unique_labels = set(labels)
print("Unique labels in the dataset:", unique_labels)

# Map labels to continuous integers if needed
label_to_id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
print("Label to ID mapping:", label_to_id)
labels = [label_to_id[label] for label in labels]

# Split dataset into train and test
train_file_paths, test_file_paths, train_labels, test_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42
)

# Define a dataset class
class SpeechDataset(Dataset):
    def __init__(self, file_paths, labels, processor):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'input_values': inputs['input_values'].squeeze(), 'labels': label}

def collate_fn(batch):
    input_values = [item['input_values'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    
    # Pad input_values to the same length
    input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=0.0)
    
    return {'input_values': input_values_padded, 'labels': labels}

# Initialize processor and model
model_name = 'facebook/wav2vec2-base-960h'
num_labels = len(label_to_id)  # Ensure num_labels matches the number of unique labels
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Create dataset instances
train_dataset = SpeechDataset(train_file_paths, train_labels, processor)
test_dataset = SpeechDataset(test_file_paths, test_labels, processor)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',  # Directory to save model and checkpoints
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = (predictions == torch.tensor(labels)).float().mean()
    return {'accuracy': accuracy.item()}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('finetuned_wav2vec2')
processor.save_pretrained('finetuned_wav2vec2')
