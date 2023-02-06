# text processing datasets
import time
import pathlib
import gzip
import json
import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Preprocessor import TextPreprocessor
from Vectorizer import TextVectorizer
from Reducer import DimensionalityReducer
from FeatureSelectors import BestSelector
from Autoencoder import Autoencoder, ConvAutoencoder
from CVAE import CVAE

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


class TextDataset:
    # Dataset initializer
    def __init__(self, path, text_col, class_col, vectorizer, seed=42):
        self.seed_ = seed

        file_extension = pathlib.Path(path).suffix
        if file_extension == '.csv':
            self.df_ = pd.read_csv(path, encoding='utf-8')
            # self.df_ = pd.read_csv(path, encoding='latin-1', header=None)
        else:
            self.df_ = self.getDF(path)

        self.df_ = self.df_.rename(columns={text_col: "text", class_col: "class"}, errors='raise')
        self.df_['text'].replace('', np.nan, inplace=True)
        self.df_.dropna(subset=['text'], inplace=True)
        # self.df_ = self.df_.iloc[:100, :]

        self.text_processor_ = TextPreprocessor(seed)
        self.preprocessing_time_ = 0
        self.vectorization_time_ = 0
        self.projection_time_ = 0
        self.dimensionality_ = 0

        # Run text preprocessing (punctuation removal, case folding, etc.).
        self.preprocess(vectorizer)

        self.df_['clean_text'].replace('', np.nan, inplace=True)
        self.df_.dropna(subset=['clean_text'], inplace=True)

        x = self.df_['clean_text']
        y = self.df_['class']

        print(x.shape)

        encoder = LabelEncoder()
        y = encoder.fit_transform(y.values)
        self.classes_ = len(list(set(y)))

        # Split the dataset into training and test sets.
        self.x_train_, self.x_test_, self.y_train_, self.y_test_ =\
            train_test_split(x, y, test_size=0.3, random_state=self.seed_, stratify=y)

        # print(self.x_train_[:100])
        # print(self.y_train_[:100])
        # enc = OneHotEncoder(drop=None)
        # self.y_train_ = enc.fit_transform(self.y_train_.values.reshape(-1, 1))
        # print(self.y_train_)

    # Apply text preprocessing of the input text
    def preprocess(self, vectorizer):
        print("Preprocessing input text... ", end="", flush=True)
        t0 = time.time()

        # If the vectorizer is word2vec, then we preprocess the text at sentence-level to capture word adjacency
        if vectorizer == "word2vec":
            self.df_['clean_text'] = self.df_.text.apply(lambda x: self.text_processor_.preprocess_sent(x))
            # print(self.df_['clean_text'].to_string())
        # otherwise, we process the text at word-level
        else:
            self.df_['clean_text'] = self.df_.text.apply(lambda x: self.text_processor_.preprocess_word(x))

        self.preprocessing_time_ = time.time() - t0
        print("completed in %5.2f sec." % self.preprocessing_time_)

    # Perform text Processing. This includes two steps: i) Text Vectorization and ii) Dimensionality Reduction.
    def process(self, vectorizer="tf-idf", reduction_method=None, latent_dim=100):
        x_train_out = x_test_out = x_train_vec = x_test_vec = None

        # ############################################################################################################
        # Step 1: Text Vectorization: Transform raw text to numerical vectors
        print("\tText Vectorization with", vectorizer, "... ", end="", flush=True)

        if vectorizer == "tf-idf" or vectorizer == "tf-idf-dmf" or vectorizer == "word2vec":
            v = TextVectorizer(algorithm=vectorizer, latent_dimensionality=latent_dim, seed=self.seed_)
            x_train_vec, x_test_vec = v.vectorize(self.x_train_, self.x_test_)
            self.vectorization_time_ = v.vectorization_time_
            self.dimensionality_ = v.get_dimensionality()

        print("completed in %5.2f sec (dimensionality = %d)." % (self.vectorization_time_, self.dimensionality_))

        # ############################################################################################################
        # Step 2: Dimensionality Reduction: Project the original data onto a reduced dimensional space.
        print("\tDimensionality reduction... ", end="", flush=True)

        # No dimensionality Reduction
        if reduction_method is None:
            x_train_out, x_test_out = x_train_vec, x_test_vec

        # Dimensionality reduction with Truncated Singular Value Decomposition
        elif reduction_method == 'tsvd':
            r = DimensionalityReducer(algorithm=reduction_method, target_space=latent_dim, seed=self.seed_)
            x_train_out, x_test_out = r.reduce(x_train_vec, x_test_vec)
            self.projection_time_ = r.reduction_time_

        # Dimensionality reduction with Non-negative Matrix Factorization
        elif reduction_method == 'nmf':
            r = DimensionalityReducer(algorithm=reduction_method, target_space=latent_dim, seed=self.seed_)
            x_train_out, x_test_out = r.reduce(x_train_vec, x_test_vec)
            self.projection_time_ = r.reduction_time_

        # Dimensionality reduction with Kernel Principal Component Analysis - RBF Kernel
        elif reduction_method == 'kpca-rbf':
            r = DimensionalityReducer(algorithm=reduction_method, target_space=latent_dim, seed=self.seed_)
            x_train_out, x_test_out = r.reduce(x_train_vec, x_test_vec)
            self.projection_time_ = r.reduction_time_

        # Dimensionality reduction with Autoencoders (remember to pass toarray() versions, otherwise tf does not work)
        elif reduction_method == 'AE':
            r = DimensionalityReducer(algorithm=reduction_method, target_space=latent_dim, seed=self.seed_)
            x_train_out, x_test_out = r.reduce(x_train_vec.toarray(), x_test_vec.toarray())
            self.projection_time_ = r.reduction_time_

        elif reduction_method == 'fs-best':
            s = BestSelector(target_space=latent_dim, measure='pearson', seed=self.seed_)
            s.fit(x_train_vec.toarray(), self.y_train_)
            x_train_out = s.transform(x_train_vec.toarray())
            x_test_out = s.transform(x_test_vec.toarray())
            self.projection_time_ = s.reduction_time_

        elif reduction_method == 'autoencoder':
            autoencoder = Autoencoder(latent_dim, self.dimensionality_)
            autoencoder.compile(optimizer='adam', loss=tf.losses.MeanSquaredError())

            autoencoder.fit(x_train_vec.toarray(), x_train_vec.toarray(), epochs=1, batch_size=16, verbose=2,
                            validation_data=(x_test_vec.toarray(), x_test_vec.toarray()))

            x_train_out = autoencoder.encoder(x_train_vec.toarray())
            x_test_out = autoencoder.encoder(x_test_vec.toarray())

        elif reduction_method == 'CAE':
            conv_autoencoder = ConvAutoencoder(latent_dim, self.dimensionality_)
            conv_autoencoder.compile(optimizer='adam', loss=tf.losses.MeanSquaredError())

            conv_autoencoder.fit(x_train_vec.toarray(), x_train_vec.toarray(), epochs=1, batch_size=16, verbose=2,
                                 validation_data=(x_test_vec.toarray(), x_test_vec.toarray()))

            x_train_out = conv_autoencoder.encoder(x_train_vec.toarray())
            x_test_out = conv_autoencoder.encoder(x_test_vec.toarray())

        elif reduction_method == 'CVAE':
            cv_autoencoder = CVAE(self.dimensionality_)
            cv_autoencoder.compile(optimizer='adam', loss=tf.losses.MeanSquaredError())

            cv_autoencoder.fit(x_train_vec.toarray(), x_train_vec.toarray(), epochs=1, batch_size=16, verbose=2,
                               validation_data=(x_test_vec.toarray(), x_test_vec.toarray()))

            x_train_out = cv_autoencoder.encoder(x_train_vec.toarray())
            x_test_out = cv_autoencoder.encoder(x_test_vec.toarray())

        print("completed in %5.2f sec." % self.projection_time_)

        return x_train_out, x_test_out, self.y_train_, self.y_test_

    # Return the dataframe
    def get_data(self):
        return self.df_

    def get_preprocessing_time(self):
        return self.preprocessing_time_

    def getDF(self, path):
        i = 0
        df = {}
        for d in self.parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    def parse(self, path):
        g = gzip.open(path, 'rb')
        for lt in g:
            yield json.loads(lt)
