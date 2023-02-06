import time
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec


class TextVectorizer:
    def __init__(self, algorithm, latent_dimensionality, seed):
        self.algorithm_ = algorithm
        self.seed_ = seed
        self.vectorization_time_ = 0
        self.dimensionality_ = 0
        self.latent_dimensionality_ = latent_dimensionality

    def vectorize(self, train_data, test_data):
        train_data_vec = test_data_vec = None

        if self.algorithm_ == 'tf-idf':
            tfidf = TfidfVectorizer()

            t0 = time.time()
            train_data_vec = tfidf.fit_transform(train_data)
            test_data_vec = tfidf.transform(test_data)

            self.vectorization_time_ = time.time() - t0
            self.dimensionality_ = len(tfidf.vocabulary_)

        elif self.algorithm_ == 'tf-idf-dmf':
            tfidf = TfidfVectorizer(max_features=15000)

            t0 = time.time()
            train_data_vec = tfidf.fit_transform(train_data)
            test_data_vec = tfidf.transform(test_data)

            self.vectorization_time_ = time.time() - t0
            self.dimensionality_ = len(tfidf.vocabulary_)

        elif self.algorithm_ == 'word2vec':
            self.dimensionality_ = self.latent_dimensionality_
            t0 = time.time()

            sentence = []
            # print("Train Data")
            # print(train_data)
            for s in train_data:
                sentence += s

            for s in test_data:
                sentence += s

            avg_sentence_length = np.average([len(s) for s in sentence])

            # print("Sentences")
            # print(sentence)

            min_count = 1
            num_processor = 8
            context = int(avg_sentence_length)
            downsampling = 0.001

            model = word2vec.Word2Vec(sentence, workers=num_processor, vector_size=self.latent_dimensionality_,
                                      min_count=min_count, window=context, sample=downsampling)
            model.init_sims(replace=True)

            # Vectorize the training data
            t_train_data_vec = []
            ctr = -1
            for t in train_data:
                ctr += 1
                t_train_data_vec.append([])
                for s in t:
                    for w in s:
                        t_train_data_vec[ctr].append(w)

            # Apply "getAvgFeatureVec" function.
            train_data_vec = self.get_avg_feature_vec(t_train_data_vec, model)

            # Vectorize the test data
            t_test_data_vec = []
            ctr = -1
            for t in test_data:
                ctr += 1
                t_test_data_vec.append([])
                for s in t:
                    for w in s:
                        t_test_data_vec[ctr].append(w)

            # Apply "getAvgFeatureVec" function.
            test_data_vec = self.get_avg_feature_vec(t_test_data_vec, model)
            # print(t_test_data_vec)
            self.vectorization_time_ = time.time() - t0

        return train_data_vec, test_data_vec

    def make_feature_vec(self, doc, model):
        feature_vec = np.zeros((self.latent_dimensionality_,), dtype="float32")

        # Unique word set
        word_index = set(model.wv.index_to_key)
        # For division, we need to count the number of words
        num_words = 0

        # Iterate words in a review and if the word is in the unique word set, add the vector values for each word.
        for word in doc:
            if word in word_index:
                num_words += 1
                feature_vec = np.add(feature_vec, model.wv[word])
            # else:
                # print("word:", word, "not in index\n")

        # if  num_words == 0:
        #    print("Zero words! in", doc)
        # Divide the sum of vector values by total number of word in a review.
        # print(feature_vec)
        if num_words > 0:
            feature_vec = np.divide(feature_vec, num_words)

        return feature_vec

    def get_avg_feature_vec(self, clean_docs, model):
        # Keep track of the sequence of reviews, create the number "th" variable.
        review_th = 0

        # Row: number of total reviews, Column: number of vector spaces (num_features, we set this in Word2Vec step).
        feature_vecs = np.zeros((len(clean_docs), self.latent_dimensionality_), dtype="float32")

        # Iterate over reviews and add the result of makeFeatureVec.
        for d in clean_docs:
            if len(d) > 0:
                vec = self.make_feature_vec(d, model)
                if vec.any():
                    feature_vecs[int(review_th)] = vec
                    # Once the vector values are added, increase the one for the review_th variable.
                    review_th += 1

        return feature_vecs

    def get_time(self):
        return self.vectorization_time_

    def get_dimensionality(self):
        return self.dimensionality_
