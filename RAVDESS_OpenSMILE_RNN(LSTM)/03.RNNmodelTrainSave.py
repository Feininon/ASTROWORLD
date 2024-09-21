import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

file_path = "iteration3/extFileNameMerged3/merged_with_emotion_intensity3.csv"
nervous_df = pd.read_csv(file_path)

# Assume X is the features and y is the label for nervousness intensity
X = nervous_df.drop(columns=['emotion_label', 'intensity_label'])
y = nervous_df['intensity_label']

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input for RNN (samples, timesteps, features)
X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

from sklearn.preprocessing import LabelEncoder

# Encode the categorical labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(nervous_df['intensity_label'])

# Convert the encoded labels to categorical (for multi-class classification)
y_categorical = to_categorical(y_encoded)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

# Build the RNN model using LSTM
model = Sequential()

# Add LSTM layers
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(units=32))
model.add(Dropout(0.3))

# Add output layer
model.add(Dense(units=y_categorical.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Save the trained model
model.save("nervous_intensity_model3.h5")
print("Model saved as 'nervous_intensity_model3.h5'")


# Load the saved model when needed
loaded_model = load_model("nervous_intensity_model3.h5")

# Use the loaded model for predictions or further training
predictions = loaded_model.predict(X_test)

