import tensorflow as tf


# The autoencoder consists of two parts: the encoder and the decoder. The encoder learns how to interpret the input
# and compress it to an internal representation defined by the bottleneck layer. The decoder takes the output of the
# encoder (the bottleneck layer) and attempts to recreate the input. Once the autoencoder is trained, the decoder is
# discarded and we keep the encoder. We use it to encode input examples to vectors output by the bottleneck layer.
class AE:
    def __init__(self, n_components, n_inputs, random_state):
        self.n_components_ = n_components
        self.n_inputs_ = n_inputs
        self.random_seed_ = random_state

        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(n_components, activation='relu'),
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(n_inputs, activation='relu')
        ])

        # Autoencoder model
        self.model = tf.keras.Sequential([
            self.encoder,
            self.decoder
        ])

    # Compile & Train the autoencoder model
    def fit_transform(self, x, xtest):
        # Compile the autoencoder model
        self.model.compile(optimizer='adam', loss='mse')

        # Fit the autoencoder model
        self.model.fit(x, x, epochs=10, batch_size=16, verbose=2, validation_data=(xtest, xtest))

        return self.transform(x)

    def transform(self, x):
        return self.encoder(x).numpy()
