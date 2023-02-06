import numpy as np
from keras import Sequential
from keras.layers import Bidirectional, LSTM, Dropout, Dense
from sklearn.preprocessing import OneHotEncoder


####################################################################################################################
# A simple BiDirectional LSTM Classifier
class LSTMClassifier:
    def __init__(self, classes, latent_dim, seed=42):
        self.labels_ = classes
        loss_function = 'binary_crossentropy'
        activation_function = 'sigmoid'
        if classes > 2:
            loss_function = 'sparse_categorical_crossentropy'
            activation_function = 'softmax'

        self.model_ = Sequential()
        # model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
        self.model_.add(LSTM(128, input_shape=(1, latent_dim), activation='relu', return_sequences=True))
        self.model_.add(Dropout(0.2))
        self.model_.add(LSTM(128, activation='relu'))
        self.model_.add(Dropout(0.2))

        # for units in [128,128,64,32]:
        # model.add(Dense(units,activation='relu'))
        # model.add(Dropout(0.2))

        self.model_.add(Dense(32, activation='relu'))
        self.model_.add(Dropout(0.2))
        self.model_.add(Dense(classes, activation=activation_function))
        self.model_.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])

        # print(self.model_.summary())

    def fit(self, x_train, y_train):
        if isinstance(x_train, np.ndarray):
            train_data_vec = x_train
        else:
            train_data_vec = x_train.toarray()

        train_data = train_data_vec.reshape(train_data_vec.shape[0], 1, train_data_vec.shape[-1])

        # One hot encode the class labels
        if self.labels_ == 2:
            ohe = OneHotEncoder()
            training_label_seq = ohe.fit_transform(y_train.reshape(-1, 1)).toarray()
        else:
            training_label_seq = y_train.reshape(-1, 1)

        self.model_.fit(train_data, training_label_seq, epochs=20, batch_size=256)

    # Use the model to make predictions on the test set
    def predict(self, x_test):
        if isinstance(x_test, np.ndarray):
            test_data_vec = x_test
        else:
            test_data_vec = x_test.toarray()

        test_data = test_data_vec.reshape(test_data_vec.shape[0], 1, test_data_vec.shape[-1])

        y_prob = self.model_.predict(test_data)
        y_class = y_prob.argmax(axis=-1)

        return y_class

    # Evaluate the model by using the Keras evaluate method
    def eval(self, x_test, y_test):
        if isinstance(x_test, np.ndarray):
            test_data_vec = x_test
        else:
            test_data_vec = x_test.toarray()

        test_data = test_data_vec.reshape(test_data_vec.shape[0], 1, test_data_vec.shape[-1])

        if self.labels_ == 2:
            ohe = OneHotEncoder()
            testing_label_seq = ohe.fit_transform(y_test.reshape(-1, 1)).toarray()
        else:
            testing_label_seq = y_test.reshape(-1, 1)

        results = self.model_.evaluate(test_data, testing_label_seq)
        print(results)


####################################################################################################################
# A simple BiDirectional LSTM Classifier
class BiLSTMClassifier:
    def __init__(self, classes, latent_dim, seed=42):
        self.labels_ = classes

        loss_function = 'binary_crossentropy'
        activation_function = 'sigmoid'
        if classes > 2:
            loss_function = 'sparse_categorical_crossentropy'
            activation_function = 'softmax'

        self.model_ = Sequential()
        # self.model_.add(Bidirectional(LSTM(latent_dim, return_sequences=False), input_shape=(1, latent_dim)))
        self.model_.add(Bidirectional(LSTM(100, return_sequences=False), input_shape=(1, latent_dim)))
        self.model_.add(Dropout(0.3, seed=seed))
        self.model_.add(Dense(classes, activation=activation_function))

        self.model_.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])

        # print(self.model_.summary())

    # Train the model
    def fit(self, x_train, y_train):
        if isinstance(x_train, np.ndarray):
            train_data_vec = x_train
        else:
            train_data_vec = x_train.toarray()

        train_data = train_data_vec.reshape(train_data_vec.shape[0], 1, train_data_vec.shape[-1])

        # One hot encode the class labels
        if self.labels_ == 2:
            ohe = OneHotEncoder()
            training_label_seq = ohe.fit_transform(y_train.reshape(-1, 1)).toarray()
        else:
            training_label_seq = y_train.reshape(-1, 1)

        self.model_.fit(train_data, training_label_seq, epochs=20, batch_size=256)

    # Use the model to make predictions on the test set
    def predict(self, x_test):
        if isinstance(x_test, np.ndarray):
            test_data_vec = x_test
        else:
            test_data_vec = x_test.toarray()

        test_data = test_data_vec.reshape(test_data_vec.shape[0], 1, test_data_vec.shape[-1])

        y_prob = self.model_.predict(test_data)
        y_class = y_prob.argmax(axis=-1)

        return y_class

    # Evaluate the model by using the Keras evaluate method
    def eval(self, x_test, y_test):
        if isinstance(x_test, np.ndarray):
            test_data_vec = x_test
        else:
            test_data_vec = x_test.toarray()

        test_data = test_data_vec.reshape(test_data_vec.shape[0], 1, test_data_vec.shape[-1])

        if self.labels_ == 2:
            ohe = OneHotEncoder()
            testing_label_seq = ohe.fit_transform(y_test.reshape(-1, 1)).toarray()
        else:
            testing_label_seq = y_test.reshape(-1, 1)

        results = self.model_.evaluate(test_data, testing_label_seq)
        print(results)
