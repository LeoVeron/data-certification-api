
from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint
@app.get("/predict/")
def predict(acousticness,
        danceability,
        duration_ms,
        energy,
        explicit,
        id,
        instrumentalness,
        key,   
        liveness,
        loudness,
        mode,
        name,
        release_date,
        speechiness,
        tempo,   
        valence,
        artist):


    dico_X=dict(acousticness=[float(acousticness)],
                          danceability=[float(danceability)],
                          duration_ms=[int(duration_ms)],
                          energy=[float(energy)],
                          explicit=[int(explicit)],
                          id=id,
                          instrumentalness=[float(instrumentalness)],
                          key=[int(key)],
                          liveness=[float(liveness)],
                          loudness=[float(loudness)],
                          mode=[int(mode)],
                          name=name,
                          release_date=release_date,
                          speechiness=[float(speechiness)],
                          tempo=[float(tempo)],
                          valence=[float(valence)],
                          artist=artist)
    X = pd.DataFrame(dico_X)
    pipeline = joblib.load('model.joblib')
    res = pipeline.predict(X)

    return dict(
        artist=artist,
        name=name,
        popularity_predicted=int(res[0]))