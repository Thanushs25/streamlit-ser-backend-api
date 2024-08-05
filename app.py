from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import librosa 
from tensorflow import keras
from keras.models import load_model
import io

# Load the trained model
model = load_model('emotionclassifier.h5')

# Define emotions corresponding to model outputs
emotions =  ['angry', 'fear', 'happy', 'neutral', 'sad']

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def hello():
    return 'route to /predict and upload the audio file to predict the emotion'

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the audio file
    contents = await file.read()
    original_features = []
    audio, sr = librosa.load(io.BytesIO(contents), sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    original_features.append(mfccs_mean)
    x = np.array(original_features)
    x = np.expand_dims(x, -1)
    # Predict emotion
    prediction = model.predict(x)
    predicted_emotion = emotions[prediction.argmax(axis=1)[0]]
    return JSONResponse(content={"emotion": predicted_emotion})

if __name__ == "__main__":
    uvicorn.run(app)
