from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

model = load_model('emotionclassifier.h5')
emotions = ['angry', 'fear', 'happy', 'neutral', 'sad']
app = FastAPI()

@app.get("/")
async def hello():
    return 'route to /predict and upload the audio file to predict the emotion'

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    audio, sr = librosa.load(io.BytesIO(contents), sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    x = np.expand_dims([mfccs_mean], -1)
    prediction = model.predict(x)
    predicted_emotion = emotions[prediction.argmax(axis=1)[0]]
    return JSONResponse(content={"emotion": predicted_emotion})
