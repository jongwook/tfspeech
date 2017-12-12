from keras.models import Sequential
from keras.layers import *
from kapre.time_frequency import Spectrogram, Melspectrogram

def spectrogram_cnn(num_classes=31, input_length=16000, mel=False):
    """
    31-class classifier based on convolutional neural network

    input_size: (None, input_length)
    output_size: (None, num_classes)
    """
    model = Sequential()
    model.add(Reshape(target_shape=(1, input_length), input_shape=(input_length,)))

    if mel:
        model.add(Melspectrogram(sr=16000, n_mels=32, return_decibel_melgram=True, n_dft=512, n_hop=128))
    else:
        model.add(Spectrogram(n_dft=512, n_hop=128, return_decibel_spectrogram=True))


    # conv 1
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='layer1_conv'))
    model.add(BatchNormalization(name='layer1_bn'))
    model.add(Dropout(0.25, name='layer1_dropout'))
    model.add(MaxPool2D(pool_size=(2,2), name='layer1_maxpool'))

    # conv 2
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='layer2_conv'))
    model.add(BatchNormalization(name='layer2_bn'))
    model.add(Dropout(0.25, name='layer2_dropout'))
    model.add(MaxPool2D(pool_size=(2,2), name='layer2_maxpool'))

    # conv 3
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='layer3_conv'))
    model.add(BatchNormalization(name='layer3_bn'))
    model.add(Dropout(0.25, name='layer3_dropout'))
    model.add(MaxPool2D(pool_size=(2,2), name='layer3_maxpool'))

    # conv 4
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='layer4_conv'))
    model.add(BatchNormalization(name='layer4_bn'))
    model.add(Dropout(0.25, name='layer4_dropout'))
    model.add(MaxPool2D(pool_size=(2,2), name='layer4_maxpool'))

    # conv 5
    if not mel:
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='layer5_conv'))
        model.add(BatchNormalization(name='layer5_bn'))
        model.add(Dropout(0.25, name='layer5_dropout'))
        model.add(MaxPool2D(pool_size=(2,2), name='layer5_maxpool'))

    # dense
    model.add(Flatten(name='layer6_flatten'))
    # model.add(Dense(128, activation='relu', name='layer6_dense'))
    # model.add(BatchNormalization(name='layer6_bn'))
    # model.add(Dropout(0.25, name='layer6_dropout'))

    # output
    model.add(Dense(num_classes, activation='softmax', name='output_softmax'))

    return model


def spectrogram_lstm(num_classes=31, input_length=16000, mel=False):
    """
        31-class classifier based on 1D convolutional neural network using LSTM

        input_size: (None, input_length)
        output_size: (None, num_classes)
    """
    model = Sequential()
    model.add(Reshape(target_shape=(1, input_length), input_shape=(input_length,)))

    if mel:
        model.add(Melspectrogram(sr=16000, n_mels=32, return_decibel_melgram=True, n_dft=512, n_hop=128))
    else:
        model.add(Spectrogram(n_dft=512, n_hop=128, return_decibel_spectrogram=True))

    model.add(Conv2D(64, kernel_size=(9, 1), activation='relu', padding='same', name='layer1_conv'))
    model.add(BatchNormalization(name='layer1_bn'))
    model.add(Dropout(0.25, name='layer1_dropout'))
    model.add(MaxPool2D(pool_size=(2, 1), name='layer4_maxpool'))

    model.add(Conv2D(64, kernel_size=(9, 1), activation='relu', padding='same', name='layer2_conv'))
    model.add(BatchNormalization(name='layer2_bn'))
    model.add(Dropout(0.25, name='layer2_dropout'))
    model.add(MaxPool2D(pool_size=(2,1), name='layer2_maxpool'))

    model.add(Conv2D(32, kernel_size=(9, 1), activation='relu', padding='same', name='layer3_conv'))
    model.add(BatchNormalization(name='layer3_bn'))
    model.add(Dropout(0.25, name='layer3_dropout'))
    model.add(MaxPool2D(pool_size=(2, 1), name='layer4_maxpool'))

    model.add(Conv2D(32, kernel_size=(9, 1), activation='relu', padding='same', name='layer4_conv'))
    model.add(BatchNormalization(name='layer4_bn'))
    model.add(Dropout(0.25, name='layer4_dropout'))
    model.add(MaxPool2D(pool_size=(2,1), name='layer4_maxpool'))

    model.add(Conv2D(32, kernel_size=(9, 1), activation='relu', padding='same', name='layer5_conv'))
    model.add(BatchNormalization(name='layer5_bn'))
    model.add(Dropout(0.25, name='layer3_dropout'))
    model.add(MaxPool2D(pool_size=(2, 1), name='layer4_maxpool'))

    model.add(Conv2D(32, kernel_size=(9, 1), activation='relu', padding='same', name='layer6_conv'))
    model.add(BatchNormalization(name='layer6_bn'))
    model.add(Dropout(0.25, name='layer6_dropout'))

    # # lstm
    model.add(Permute((1, 3, 2), name='layer7_permute'))
    shape = model.output_shape

    model.add(Reshape((shape[1] * shape[2], shape[3]), name='layer7_reshape'))
    model.add(LSTM(64, return_sequences=True, name='layer6_lstm'))
    model.add(LSTM(64, name='layer7_lstm'))
    model.add(Dense(num_classes, activation='softmax', name='output_softmax'))

    return model
