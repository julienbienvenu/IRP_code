
from matplotlib import pyplot as plt
import numpy as np


def test_model(autoencoder, X_test, y_test):

    test_images = X_test

    # Reconstruct test images using the autoencoder
    reconstructed_images = autoencoder.predict(test_images)

    # Calculate the reconstruction loss
    mse = np.mean(np.power(test_images - np.squeeze(reconstructed_images), 2), axis=(1, 2))

    mean_mse = np.mean(mse)

    # Print the mean reconstruction loss
    print("Mean Reconstruction Loss:", mean_mse)

    # Plot some test images and their reconstructed versions
    n = min(len(X_test), 5)  # number of images to plot
    plt.figure(figsize=(20, 10))

    for i in range(n):
        # Plot original test image
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(test_images[i], cmap = 'gray_r')
        plt.title("Original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Plot reconstructed image
        ax = plt.subplot(3, n, i + n + 1)
        plt.imshow(reconstructed_images[i], cmap = 'gray_r')
        plt.title(f"Reconstructed : {np.count_nonzero(reconstructed_images[i])}")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Objective
        ax = plt.subplot(3, n, i + 2*n + 1)
        plt.imshow(y_test[i], cmap = 'gray_r')
        plt.title(f"Objective : {np.count_nonzero(y_test[i])}")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig('plot/validation/reconstructed_image.png', dpi=300)
    plt.clf()

def plot_results(history):

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plot/validation/loss.png')
    plt.clf()

    plt.plot(history.history['accuracy_00'])
    plt.plot(history.history['val_accuracy_00'])
    plt.title('Model Acccuracy 00')
    plt.ylabel('Accuracy00')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plot/validation/accuracy00.png')
    plt.clf()

    plt.plot(history.history['accuracy_0'])
    plt.plot(history.history['val_accuracy_0'])
    plt.title('Model Acccuracy 0')
    plt.ylabel('Accuracy0')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plot/validation/accuracy0.png')
    plt.clf()

    plt.plot(history.history['accuracy_1'])
    plt.plot(history.history['val_accuracy_1'])
    plt.title('Model Acccuracy 1')
    plt.ylabel('Accuracy 1')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plot/validation/accuracy1.png')
    plt.clf()