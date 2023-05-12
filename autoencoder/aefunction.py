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

# class Autoencoder(tf.keras.Model):
#     def __init__(self, d0, k, p):
#         super(Autoencoder, self).__init__()
#         self.d0 = d0
#         self.k = k
#         self.p = p

#         # Encoder layers
#         self.encoder_conv1 = tf.keras.layers.Conv2D(64, k, padding='same')
#         self.encoder_leakyrelu1 = tf.keras.layers.LeakyReLU()
#         self.encoder_simagpool1 = tf.keras.layers.MaxPooling2D(pool_size=(1,2), strides=(1,2))
#         self.encoder_conv2 = tf.keras.layers.Conv2D(64, k, padding='same')
#         self.encoder_leakyrelu2 = tf.keras.layers.LeakyReLU()
#         self.encoder_simagpool2 = tf.keras.layers.MaxPooling2D(pool_size=(1,2), strides=(1,2))
#         self.encoder_conv3 = tf.keras.layers.Conv2D(64, k, padding='same')
#         self.encoder_leakyrelu3 = tf.keras.layers.LeakyReLU()
#         self.encoder_simagpool3 = tf.keras.layers.MaxPooling2D(pool_size=(1,2), strides=(1,2))

#         # Decoder layers
#         self.decoder_unpool1 = tf.keras.layers.UpSampling2D(size=(1,2))
#         self.decoder_relu1 = tf.keras.layers.ReLU()
#         self.decoder_unpool2 = tf.keras.layers.UpSampling2D(size=(1,2))
#         self.decoder_relu2 = tf.keras.layers.ReLU()
#         self.decoder_unpool3 = tf.keras.layers.UpSampling2D(size=(1,2))
#         self.decoder_relu3 = tf.keras.layers.ReLU()
#         self.decoder_conv = tf.keras.layers.Conv2D(d0, k, padding='same')

#     def encode(self, x):
#         x = tf.reshape(x, [-1, 10000, 1, 1])
#         x = self.encoder_conv1(x)
#         x = self.encoder_leakyrelu1(x)
#         x = self.encoder_simagpool1(x)
#         x = self.encoder_conv2(x)
#         x = self.encoder_leakyrelu2(x)
#         x = self.encoder_simagpool2(x)
#         x = self.encoder_conv3(x)
#         x = self.encoder_leakyrelu3(x)
#         x = self.encoder_simagpool3(x)
#         return x

#     def decode(self, x):
#         x = self.decoder_unpool1(x)
#         x = self.decoder_relu1(x)
#         x = self.decoder_unpool2(x)
#         x = self.decoder_relu2(x)
#         x = self.decoder_unpool3(x)
#         x = self.decoder_relu3(x)
#         x = self.decoder_conv(x)
#         return x

#     def call(self, x):
#         encoded = self.encode(x)
#         decoded = self.decode(encoded)
#         return decoded

def accuracy(y_true, y_pred):

    # Flatten y_true and y_pred to compare each element
    y_true_flat = tf.keras.layers.Flatten()(y_true)
    y_pred_flat = tf.keras.layers.Flatten()(y_pred)

    y_pred_flat = tf.squeeze(y_pred, axis=-1)

    # Count the number of correct 1s
    correct_ones = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true_flat, 1), tf.equal(tf.round(y_pred_flat), 1)), dtype=tf.float32))

    # Compute the total number of 1s in y_true
    total_ones = tf.reduce_sum(tf.cast(tf.equal(y_true_flat, 1), dtype=tf.float32))

    # Compute the accuracy as the ratio of correct 1s to total 1s
    acc = correct_ones / total_ones

    return acc

def loss(y_true, y_pred):
    true_sum = tf.reduce_sum(y_true)
    pred_sum = tf.reduce_sum(y_pred)
    loss = tf.abs(true_sum - pred_sum)
    return loss

def define_autoencoder():
    input_shape = (100, 100, 1)
    autoencoder = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')
    ])

    return autoencoder

def run_model(autoencoder, X_train, X_test, y_train, y_test, epochs=20):

    history = History()
    early_stop = EarlyStopping(monitor='loss', patience=25)

    print(f'Dataset : Train {len(X_train)}/{len(y_train)}, Test {len(X_test)}/{len(y_test)}')

    autoencoder.compile(optimizer='adam', loss=loss, metrics=[accuracy])
    autoencoder.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose = 1, validation_data=(X_test, y_test), callbacks=[history, early_stop])
    
    save_model(autoencoder, 'autoencoder/autoencoder_model.h5')

    return autoencoder, history

def evaluate_autoencoder():

    return 0