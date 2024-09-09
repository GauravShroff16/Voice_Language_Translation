import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# load the whisper model and processor
model_name = "openai/whisper-large"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def translate_audio(audio_path):
    # load and preprocess audio
    audio, _ = librosa.load(audio_path, sr=16000)  # whisper model needs 16kHz audio so convert it or use direct
    
    # process the audio to the format the model needs
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    
    # extract the input features from the processed inputs
    audio_input = inputs['input_features'].to(device)
    
    # make sure attention mask is included
    attention_mask = inputs.get('attention_mask')
    if attention_mask is None:
        attention_mask = torch.ones(audio_input.shape, dtype=torch.long, device=device)
    
    # get forced decoder IDs for translation to english
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="translate")
    
    # generate transcription with translation to english
    with torch.no_grad():
        generated_ids = model.generate(
            audio_input,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_decoder_ids
        )
    
    # decode the generated tokens
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

if name == "main":
    audio_path = r"C:\Users\madhu\OneDrive\Desktop\spanishh.wav"
    translated_text = translate_audio(audio_path)
    print("Translated Text:", translated_text)
