# This function is the main function

import numpy as np
from autoencoder.aefunction import define_autoencoder, run_model
from autoencoder.model_validation import plot_results, test_model
from dataset.generate_dataset import generate_tif_dataset


def main():

    # Generate dataset
    X_train, X_test, y_train, y_test = generate_tif_dataset(max_values = 100) 
    
    # Load and compile the model
    size = X_train[0].shape[0]
    autoencoder = define_autoencoder()

    # Run the model
    autoencoder, history = run_model(autoencoder, X_train, X_test, y_train, y_test, epochs = 500)

    # Retrieve the resuls
    test_model(autoencoder, X_test, y_test)
    plot_results(history)

if __name__ == '__main__':

    main()