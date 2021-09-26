from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Reshape, Flatten, Dense, BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2DTranspose

class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = Sequential()
        self.encoder.add(Input(shape=(28, 28, 1)))
        self.encoder.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.encoder.add(MaxPool2D((2, 2), padding='same'))
        self.encoder.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.encoder.add(MaxPool2D((2, 2), padding='same'))
        self.encoder.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.encoder.add(Flatten())
        self.encoder.add(Dense(49, activation='softmax'))

        self.decoder = Sequential()
        self.decoder.add(Reshape((7, 7, 1)))
        self.decoder.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        self.decoder.add(BatchNormalization())
        self.decoder.add(Conv2DTranspose(32, (3, 3), padding='same', activation='relu'))
        self.decoder.add(Conv2D(1, (3, 3), padding='same', activation='sigmoid'))

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded