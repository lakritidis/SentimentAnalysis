import time
import numpy as np
from sklearn.feature_selection import mutual_info_classif


class BestSelector:
    def __init__(self, target_space, measure, seed=42):
        self.latent_dim_ = target_space
        self.measure_ = measure
        self.seed_ = seed
        self.best_features_ = []
        self.reduction_time_ = 0

    def fit(self, features, labels):
        t0 = time.time()

        feature_scores = {}
        num_features = features.shape[1]
        for f in range(num_features):
            # print("Fitting ", f)
            feat = features[:, f].reshape(-1, 1)
            score = mutual_info_classif(feat, labels, discrete_features=[False], random_state=self.seed_)
            feature_scores[f] = score[0]

        value_key_pairs = ((value, key) for (key, value) in feature_scores.items())
        sorted_feature_scores = sorted(value_key_pairs, reverse=True)

        self.best_features_ = [value for (key, value) in sorted_feature_scores]
        self.best_features_ = self.best_features_[: self.latent_dim_]
        self.reduction_time_ = time.time() - t0

    def transform(self, data):
        t0 = time.time()
        num_features = data.shape[1]
        delete_columns = [feat for feat in range(num_features) if feat not in self.best_features_]

        self.reduction_time_ += time.time() - t0
        return np.delete(data, delete_columns, axis=1)

    def fit_transform(self, features, labels):
        self.fit(features, labels)
        return self.transform(features)

    def get_time(self):
        return self.reduction_time_
