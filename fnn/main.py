from microns.main import scan
from numpy import full, concatenate
import pandas as pd
import os

# load the model and neuron ids of the MICrONS scan 8-5
model, ids = scan(session=8, scan_idx=5)

print(f"model type: {type(model)}")  #model is fnn.model.networks.Visual
print(f"ids type: {type(ids)}, with shape {ids.shape}")  # ids are dataframe with shape (9941, 3)

##adding this to inspect their input data
#print(ids.head())

# example 3-second video (3 x 30 frames @ 30 FPS, 144 height, 256 width)
frames = concatenate([
    full(shape=[30, 144, 256], dtype="uint8", fill_value=0),  # 1 second of black
    full(shape=[30, 144, 256], dtype="uint8", fill_value=128),  # 1 second of gray
    full(shape=[30, 144, 256], dtype="uint8", fill_value=255),  # 1 second of white
])

# predict the response of neurons to the 3-second video
#response = model.predict(stimuli=frames)
#print(response)
