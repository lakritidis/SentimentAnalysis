import tensorflow as tf


# Implementing a simple Autoencoder architecture as a subclass of tf.keras.Model superclass.
# https://www.tensorflow.org/tutorials/generative/autoencoder
class Autoencoder(tf.keras.Model):
    def __init__(self, latent_dim, original_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_dim, activation='relu')
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(original_dim, activation='relu'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ConvAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim, original_dim):
        super(ConvAutoencoder, self).__init__()
        self.embedding_dim = 20
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=10, output_dim=self.embedding_dim, input_length=2),
            tf.keras.layers.Conv1D(latent_dim, kernel_size=3, activation='relu', padding='same')
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv1DTranspose(original_dim, kernel_size=3, activation='relu', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

'''
class ConvAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim, original_dim):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
'''

class AnomalyDetector(tf.keras.Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu")])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(140, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
