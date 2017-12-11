import time

import keras
from keras.callbacks import ModelCheckpoint

import models
import data

def train():
    model = models.spectrogram_cnn(mel=True)
    train, validation, test = data.load()

    model.summary()

    checkpointer = ModelCheckpoint(filepath='/tmp/tfspeech-%d.hdf5' % int(time.time()), verbose=1, save_best_only=True)

    model.fit(
        x=train[0], y=train[1],
        batch_size=64, epochs=100, callbacks=[checkpointer],
        validation_data = validation
    )

if __name__ == "__main__":
    train()
