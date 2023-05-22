from keras.models import load_model
from autoencoder.model_validation import test_model
from autoencoder.aefunction import loss, accuracy

from dataset.generate_dataset import generate_tif_dataset

def test():
    model = load_model('autoencoder/autoencoder_model.h5', custom_objects={'loss': loss, 'accuracy' : accuracy})

    # Generate dataset
    X_train, X_test, y_train, y_test = generate_tif_dataset(max_values = 100)

    test_model(model, X_test, y_test)

if __name__ == '__main__':

    test()