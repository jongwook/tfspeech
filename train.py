import keras
import models
import data

def train():
    model = models.spectrogram_cnn()
    train, validation, test = data.load()

    model.summary()

    model.fit(
        x=train[0], y=train[1],
        batch_size=128, epochs=100, callbacks=None,
        validation_data = validation
    )

if __name__ == "__main__":
    train()
