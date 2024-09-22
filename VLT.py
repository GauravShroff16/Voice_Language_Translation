import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import uvicorn

# Load the whisper model and processor
model_name = "openai/whisper-large"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize FastAPI
app = FastAPI()

# Function for audio translation
def translate_audio(audio):
    # Process the audio to the format the model needs
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    
    # Extract the input features from the processed inputs
    audio_input = inputs['input_features'].to(device)
    
    # Get forced decoder IDs for translation to English
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="translate")
    
    # Generate transcription with translation to English
    with torch.no_grad():
        generated_ids = model.generate(
            audio_input,
            forced_decoder_ids=forced_decoder_ids
        )
    
    # Decode the generated tokens
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# Serve an HTML form for uploading audio files
@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <html>
        <head>
            <style>
                body {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    font-family: 'Pacifico', cursive;
                    background-image: url('https://wallpaperaccess.com/full/277647.jpg');
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    opacity: 0.8;
                    border: 10px double #96ACA0;
                    padding: 20px;
                    box-sizing: border-box;
                    height: 100vh;
                }
                .oval {
                    background-color: rgba(249, 200, 194, 0.8);
                    padding: 20px;
                    border-radius: 50px;
                    display: inline-block;
                    margin: 20px auto;
                }
                h2 {
                    font-size: 36px;
                    color: black;
                    margin: 0;
                }
                p {
                    font-family: 'Dancing Script', cursive;
                    color: black;
                    font-size: 32px;
                    margin-top: 20px;
                    text-align: center;
                }
                .container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    margin-top: 50px;
                    width: 100%;
                }
                .upload-form {
                    text-align: center;
                    margin-bottom: 50px;
                }
                .subheading {
                    font-family: 'Pacifico', cursive;
                    font-size: 28px;
                    color: black;
                    margin-top: 20px;
                }
                input[type="file"] {
                    display: none;
                }
                .file-label {
                    background-color: #c2fbd7;
                    border-radius: 100px;
                    cursor: pointer;
                    display: inline-block;
                    font-family: CerebriSans-Regular, -apple-system, system-ui, Roboto, sans-serif;
                    padding: 7px 20px;
                    text-align: center;
                    margin-bottom: 30px;
                }
                .file-label:hover {
                    box-shadow: rgba(44, 187, 99, .35) 0 -25px 18px -14px inset;
                    transform: scale(1.05) rotate(-1deg);
                }
                .button-container {
                    position: relative;
                    margin-top: 50px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .button-background {
                    position: absolute;
                    width: 450px;
                    height: 450px;
                    background-image: url('https://i.pinimg.com/564x/5d/e9/a1/5de9a128c664fb87dddc79feda71e0c8.jpg');
                    background-size: contain;
                    background-repeat: no-repeat;
                    border-radius: 100px;
                    z-index: -1;
                    margin-top: 250px;
                }
                .button-53 {
                    background-color: #F9C8C2;
                    color: #000000;
                    display: flex;
                    font-size: 1rem;
                    font-weight: 700;
                    justify-content: center;
                    padding: .75rem 1.65rem;
                    max-width: 460px;
                    cursor: pointer;
                    position: relative;
                    z-index: 1;
                }
                .button-53:focus {
                    outline: 0;
                }
                @media (min-width: 768px) {
                    .button-53 {
                        padding: .75rem 3rem;
                        font-size: 1.25rem;
                    }
                }
            </style>
            <link href="https://fonts.googleapis.com/css2?family=Dancing+Script&display=swap" rel="stylesheet">
            <link href="https://fonts.googleapis.com/css2?family=Pacifico&display=swap" rel="stylesheet">
        </head>
        <body>
        <div class="oval">
            <h2>Voice Translation System</h2>
        </div>
        <p>This is a multi-language translation system which is made using artificial intelligence which can translate up to 99 languages.</p>
        
        <div class="container">
            <div class="upload-form">
                <div class="subheading">Upload the audio file that needs to be translated:</div>
                <form action="/uploadfile/" enctype="multipart/form-data" method="post">
                    <label class="file-label">
                        Choose File
                        <input name="file" type="file" accept="audio/*" required>
                    </label>
                    <div class="button-container">
                        <div class="button-background"></div>
                        <button class="button-53" type="submit">Translate</button>
                    </div>
                </form>
            </div>
        </div>
        </body>
    </html>
    """
    return content


# Handle file upload
@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    # Load the uploaded file
    audio, _ = librosa.load(file.file, sr=16000)  # Convert the audio file to 16kHz for Whisper
    
    # Translate the audio using the model
    translated_text = translate_audio(audio)
    
    # Serve the HTML with the translated text
    content = f"""
    <html>
        <head>
            <link href="https://fonts.googleapis.com/css2?family=Dancing+Script&family=Pacifico&display=swap" rel="stylesheet">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-image: url('https://wallpaperaccess.com/full/277647.jpg');
                    background-size: cover;
                    background-position: center;
                    text-align: center;
                    padding: 50px;
                    color: white;
                }}
                h2 {{
                    font-family: 'Pacifico', cursive;
                    color: black;
                }}
                .output-container {{
                    display: flex;
                    align-items: center;
                    margin-top: 20px;
                }}
                .output {{
                    font-family: 'Dancing Script', cursive;
                    color: #8e8b63;
                    font-size: 36px;
                    margin-left: 20px;
                    padding: 20px;
                    border: 2px solid #C69491;
                    border-radius: 20px;
                    background-color: #eddada;
                    display: inline-block;
                }}
                .gif {{
                    width: 50px;
                    height: auto;
                }}
                .button-container {{
                    display: flex;
                    justify-content: flex-end;
                    margin-top: 20px;
                }}
                .button-53 {{
                    background-color: rgba(249, 200, 194, 0.8);
                    border: 0 solid #E5E7EB;
                    box-sizing: border-box;
                    color: #000000;
                    display: flex;
                    font-size: 1rem;
                    font-weight: 700;
                    justify-content: center;
                    line-height: 1.75rem;
                    padding: .75rem 1.65rem;
                    cursor: pointer;
                    border-radius: 5px;
                    transition: background-color 0.3s;
                }}
                .button-53:hover {{
                    background-color: rgba(249, 200, 194, 1);
                }}
            </style>
        </head>
        <body>
            <h2>Translated Text</h2>
            <div class="output-container">
                <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExcTdsYm1sNGpubjg4YzFob2FxcWFtcWV5d3p6Z3ViZjZvaGV4dDFxaSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/RBbcXmt0syqw10MBQy/giphy.webp" alt="Loading GIF" class="gif">
                <div class="output">{translated_text}</div>
            </div>
            <div class="button-container">
                <form action="/" method="get">
                    <button class="button-53" type="submit">Translate Another File</button>
                </form>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

# To run the app, use:
# uvicorn script_name:app --reload
