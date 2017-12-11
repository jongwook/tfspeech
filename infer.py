import keras
import sys
import numpy as np

# yes, no, up, down, left, right, on, off, stop, go
mapping = {
    "_background_noise_": "silence",
    "bed": "unknown",
    "bird": "unknown",
    "cat": "unknown",
    "dog": "unknown",
    "down": "down",
    "eight": "unknown",
    "five": "unknown",
    "four": "unknown",
    "go": "go",
    "happy": "unknown",
    "house": "unknown",
    "left": "left",
    "marvin": "unknown",
    "nine": "unknown",
    "no": "no",
    "off": "off",
    "on": "on",
    "one": "unknown",
    "right": "right",
    "seven": "unknown",
    "sheila": "unknown",
    "six": "unknown",
    "stop": "stop",
    "three": "unknown",
    "tree": "unknown",
    "two": "unknown",
    "up": "unknown",
    "wow": "unknown",
    "yes": "yes",
    "zero": "unknown"
}

sorted_keys = list(mapping.keys())
sorted_keys.sort()

string_for_label = [mapping[s] for s in sorted_keys]



print("Model: ", sys.argv[1])

model = keras.load(sys.argv[1])
data_path = 'data/preprocessed/test.npy'

print("Loading test data...")
data = np.load(data_path)

print("Test data shape:", data.shape)

print("Predicting...")
prediction = model.predict(data, batch_size=64, verbose=1)
prediction = np.argmax(prediction, axis=1)

filenames = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
filenames.sort()

with open("submission.csv") as f:
    print("fname,label", file=f)
    for i in len(filenames):
        print("%s,%s" % (filenames[i], string_for_label[prediction[i]]))
