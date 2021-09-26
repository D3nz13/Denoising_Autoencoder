import numpy as np
import matplotlib.pyplot as plt
from DenoisingAutoencoder import DenoisingAutoencoder
from tensorflow.keras.datasets.mnist import load_data

def get_data(noise_factor=0.4):
    """
    This function loads the data, normalizes it and adds noise. Returns both clean and noisy sets
    """
    (X_train, _), (X_test, _) = load_data()

    X_train = X_train/255
    X_test = X_test/255

    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

    X_train_noisy = np.clip(X_train_noisy, 0, 1)
    X_test_noisy = np.clip(X_test_noisy, 0, 1)

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
