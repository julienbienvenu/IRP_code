import tensorflow as tf
from keras.callbacks import History
from keras.models import save_model
from keras.callbacks import EarlyStopping

'''
Autoencoder :
- d0 is the number of input dimensions (in this case, 100*100=10000),
- k is the kernel size for the convolutional layers, 
- p is the pooling factor for the SimAGpool layers. 
'''
def accuracy_1(y_true, y_pred):

    # Count the number of correct 1s
    # correct_ones = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_flat, 1), tf.equal(tf.round(y_pred_flat), 1)), dtype=tf.float32))
    
    # Flatten the tensor
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    # Count the total number of ones
    correct_ones = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_flat, 0.5), tf.equal(tf.round(y_pred_flat), 0.5)), dtype=tf.float32))
    total_ones = tf.reduce_sum(tf.cast(tf.equal(y_true_flat, 0.5), dtype=tf.float32))

    # Print the result
    return (correct_ones / total_ones)

def accuracy_0(y_true, y_pred):

    # Count the number of correct 1s
    # correct_ones = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_flat, 1), tf.equal(tf.round(y_pred_flat), 1)), dtype=tf.float32))
    
    # Flatten the tensor
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    # Count the total number of ones
    correct_ones = tf.reduce_sum(tf.cast(tf.logical_and(tf.greater(y_true_flat, 0.25), tf.greater(y_pred_flat, 0.25)), dtype=tf.float32))
    total_ones = tf.reduce_sum(tf.cast(tf.equal(y_true_flat, 0.0), dtype=tf.float32))

    # Print the result
    return (correct_ones / total_ones)

def accuracy_00(y_true, y_pred):

    # Count the number of correct 1s
    # correct_ones = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_flat, 1), tf.equal(tf.round(y_pred_flat), 1)), dtype=tf.float32))
    
    # Flatten the tensor
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    # Count the total number of ones
    correct_ones = tf.reduce_sum(tf.cast(tf.logical_and(tf.greater(y_true_flat, 0.0), tf.greater(y_pred_flat, 0.0)), dtype=tf.float32))
    total_ones = tf.reduce_sum(tf.cast(tf.equal(y_true_flat, 0.0), dtype=tf.float32))

    # Print the result
    return (correct_ones / total_ones)


def loss(y_true, y_pred):
    '''
    This function first finds the indices of the 1s in both the true and predicted grids using tf.where.
    Then, it converts the indices to float32 values using tf.cast to allow for computation of the distance. 
    The Euclidean distance between the indices is computed using tf.sqrt(tf.reduce_sum(tf.square(true_coords - pred_coords), axis=1)). 
    Finally, the function returns the absolute difference in the sum of true and predicted values along with the sum of the distances as the loss.
    '''

    true_indices = tf.where(tf.greater(y_true, 0))
    pred_indices = tf.where(tf.greater(y_pred, 0))
    true_coords = tf.cast(true_indices, tf.float32)
    pred_coords = tf.cast(pred_indices, tf.float32)
    true_sum = tf.reduce_sum(y_true)
    pred_sum = tf.reduce_sum(y_pred)
    
    # Slice pred_coords to match true_coords
    n_true_coords = tf.shape(true_coords)[0]
    n_pred_coords = tf.shape(pred_coords)[0]

    # print(n_true_coords, n_pred_coords)

    if n_true_coords > n_pred_coords:
        true_coords = tf.slice(true_coords, begin=[0, 0], size=[n_pred_coords, -1]) 
    else :
        pred_coords = tf.slice(pred_coords, begin=[0, 0], size=[n_true_coords, -1])
    
    distance = tf.sqrt(tf.reduce_sum(tf.square(true_coords - pred_coords), axis=1))
    loss = tf.abs(true_sum - pred_sum)*0.5 #+ tf.cast(tf.abs(int(n_true_coords) - int(n_pred_coords)), tf.float32) * 0.3 + tf.reduce_sum(distance) * 0.2
    
    return loss

def define_autoencoder():
    input_shape = (100, 100, 1)
    autoencoder = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),   
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')
    ])

    return autoencoder

def run_model(autoencoder, X_train, X_test, y_train, y_test, epochs=500):

    history = History()
    early_stop = EarlyStopping(monitor='loss', patience=500)
    print(f'Dataset : Train {len(X_train)}/{len(y_train)}, Test {len(X_test)}/{len(y_test)}')

    autoencoder.compile(optimizer='adam', loss='mse', metrics=[accuracy_00, accuracy_0, accuracy_1])
    autoencoder.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose = 1, validation_data=(X_test, y_test), callbacks=[history, early_stop])
    
    save_model(autoencoder, 'autoencoder/autoencoder_model.h5')

    return autoencoder, history

def evaluate_autoencoder():

    return 0