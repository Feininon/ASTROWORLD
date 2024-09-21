import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

model_path = 'finetuned_wav2vec2'
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
model.eval() 

def predict_nervousness(audio_file_path):
    waveform, sample_rate = torchaudio.load(audio_file_path)
    
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=-1).item()
    
    nervousness_scale = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: '10'}
    predicted_nervousness = nervousness_scale.get(predicted_label, 'Unknown')

    return predicted_nervousness

nervousness = predict_nervousness(r"C:\Users\hanus\Videos\programming\KCG_main\KCG_model1\resampling\03-01-01-01-01-02-24.wav")
print(f"The nervousness level is: {nervousness} out of 10")
