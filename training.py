import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.mnist import load_data
from DenoisingAutoencoder import DenoisingAutoencoder

def get_data(noise_factor=0.4):
    """
    This function loads the data, normalizes it and adds noise. Returns both clean and noisy sets
    """
    (X_train, _), (X_test, _) = load_data()

    # normalization
    X_train = X_train/255
    X_test = X_test/255

    # adding random noise
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

    X_train_noisy = np.clip(X_train_noisy, 0, 1)
    X_test_noisy = np.clip(X_test_noisy, 0, 1)
    
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_train_noisy = X_train_noisy.reshape(-1, 28, 28, 1)

    X_test = X_test.reshape(-1, 28, 28, 1)
    X_test_noisy = X_test_noisy.reshape(-1, 28, 28, 1)

    return X_train, X_train_noisy, X_test, X_test_noisy


def plot_images(X_train, X_train_noisy, num=6):
    """
    This function plots the given number of clean and noisy images
    """
    for i in range(num):
        plt.subplot(2, num, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_train[i], cmap='gray')
        plt.subplot(2, num, i + 1 + num)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_train_noisy[i], cmap='gray')
    plt.show()


def plot_history(history):
    """
    This function plots the loss value during the training
    """
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'])
    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['val_loss'])
    plt.title('Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.show()


def plot_results(X_test, X_test_noisy, X_test_encoded, X_test_decoded, num=6):
    """
    This function plots the test images, noisy images, encoded images and the denoised output
    """
    for i in range(num):
        plt.subplot(4, num, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_test[i], cmap='gray')
        plt.subplot(4, num, i + 1 + num)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_test_noisy[i], cmap='gray')
        plt.subplot(4, num, i + 1 + 2*num)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_test_encoded[i], cmap='gray')
        plt.subplot(4, num, i + 1 + 3*num)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_test_decoded[i], cmap='gray')
    plt.show()

if __name__ == '__main__':
    # getting the data
    X_train, X_train_noisy, X_test, X_test_noisy = get_data()
    plot_images(X_train, X_train_noisy)

    # building the model
    autoencoder = DenoisingAutoencoder()
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(X_train_noisy, X_train, validation_data=(X_test_noisy, X_test), batch_size=128, epochs=5)
    plot_history(history)

    # encoding the X_test_noisy
    encoded = autoencoder.encoder.predict(X_test_noisy).reshape(-1, 7, 7)

    # decoding the encoded data
    preds = autoencoder.decoder.predict(encoded)
    plot_results(X_test, X_test_noisy, encoded, preds)
    autoencoder.evaluate(X_test_noisy, X_test)

    # model summary
    autoencoder.summary()

    # saving the model
    autoencoder.save_weights('autoencoder_weights.h5')