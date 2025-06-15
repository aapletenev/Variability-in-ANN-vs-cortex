from fnn import microns
from numpy import full, concatenate

# load the model and neuron ids of the MICrONS scan 8-5
model, ids = microns.scan(session=8, scan_idx=5)

# example 3-second video (3 x 30 frames @ 30 FPS, 144 height, 256 width)
frames = concatenate([
    full(shape=[30, 144, 256], dtype="uint8", fill_value=0),   # 1 second of black
    full(shape=[30, 144, 256], dtype="uint8", fill_value=128), # 1 second of gray
    full(shape=[30, 144, 256], dtype="uint8", fill_value=255), # 1 second of white
])

# predict the response of neurons to the 3-second video
response = model.predict(stimuli=frames)
print(response)