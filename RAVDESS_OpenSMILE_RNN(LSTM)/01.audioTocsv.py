import os
import opensmile
import pandas as pd

# Initialize OpenSMILE for feature extraction (GeMAPS is commonly used for emotion detection)
# Initialize the opensmile feature extractor
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,  # Use 'emobase' for emotion-related features
    feature_level=opensmile.FeatureLevel.Functionals  # Functionals gives summarized features
)
# Directory containing the audio files
audio_dir = 'ipsamp'
output_feature_dir = 'iteration3/opscsvi3'

# Ensure output directory exists
if not os.path.exists(output_feature_dir):
    os.makedirs(output_feature_dir)


# Loop through all audio files in the folder
for audio_file in os.listdir(audio_dir):
    if audio_file.endswith('.wav'):  # Process .wav files
        input_audio_path = os.path.join(audio_dir, audio_file)
       
        # Extract features using OpenSMILE
        features = smile.process_file(input_audio_path)
       
        # Save the extracted features as CSV
        output_feature_path = os.path.join(output_feature_dir, f'{os.path.splitext(audio_file)[0]}.csv')
        features.to_csv(output_feature_path, index=False)
       
        print(f"Extracted features for {audio_file} saved to {output_feature_path}")

print("Feature extraction complete.")