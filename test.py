from keras.models import load_model
from autoencoder.model_validation import test_model
from autoencoder.aefunction import ones_count_loss, accuracy_1, accuracy_0, accuracy_00

from dataset.generate_dataset import generate_tif_dataset

def test():
    model = load_model('autoencoder/autoencoder_model_test.h5', custom_objects={'ones_count_loss': ones_count_loss, 'accuracy_1': accuracy_1, 'accuracy_0': accuracy_0, 'accuracy_00': accuracy_00})

    # Generate dataset
    X_train, X_test, y_train, y_test = generate_tif_dataset(max_values = 100)

    test_model(model, X_test, y_test)

if __name__ == '__main__':

    test()