import os
import pandas as pd

fileno=0
# Define a mapping for emotion codes and intensities
emotion_map = {
    1: 'neutral',
    2: 'calm',
    3: 'happy',
    4: 'sad',
    5: 'angry',
    6: 'fearful',
    7: 'disgust',
    8: 'surprised'
}

intensity_map = {
    1: 'normal',
    2: 'strong'
}

# Function to parse the RAVDESS filename and extract emotion and intensity
def parse_ravdess_filename(filename):
    parts = filename.split("-")
    emotion_code = int(parts[2])  # Extract emotion code
    intensity_code = int(parts[3])  # Extract intensity code
    
    emotion_label = emotion_map.get(emotion_code, 'unknown')  # Map the emotion
    intensity_label = intensity_map.get(intensity_code, 'unknown')  # Map the intensity
    
    return emotion_label, intensity_label

# Directory containing the CSV files
csv_dir = "iteration3/opscsvi3"
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

# Initialize an empty list to store the data
dataframes = []

# Loop through each CSV file and read its contents
for file in csv_files:
    file_path = os.path.join(csv_dir, file)
    
    # Read the CSV data
    df = pd.read_csv(file_path)
    
    # Extract emotion and intensity from the filename
    emotion_label, intensity_label = parse_ravdess_filename(file)
    
    # Add the emotion and intensity columns to the dataframe
    df['emotion_label'] = emotion_label
    df['intensity_label'] = intensity_label
    
    # Append the dataframe to the list
    dataframes.append(df)
    
    # nothing here just want it to look cool
    fileno=fileno+1
    print(f"appending them col to da df {fileno}")

# Concatenate all dataframes into one large dataframe
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the merged dataframe with emotion and intensity labels to a new CSV
output_file = 'iteration3\extFileNameMerged3\merged_with_emotion_intensity3.csv'
merged_df.to_csv(output_file, index=False)

print(f"Emotion and intensity labels added and saved to {output_file}")





