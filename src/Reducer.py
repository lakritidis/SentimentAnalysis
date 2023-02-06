import time
from sklearn.decomposition import TruncatedSVD, NMF, KernelPCA

from AE import AE


class DimensionalityReducer:
    def __init__(self, algorithm, target_space, seed=42):
        self.target_space_ = target_space
        self.algorithm_ = algorithm
        self.seed_ = seed
        self.reduction_time_ = 0

    def reduce(self, train_data, test_data):
        model = None

        t0 = time.time()

        # Truncated Singular Value Decomposition: Similar to PCA without data centering.
        if self.algorithm_ == 'tsvd':
            model = TruncatedSVD(n_components=self.target_space_, random_state=self.seed_)

        # Non-Negative Matrix Factorization: Find two non-negative matrices whose product approximates the non-negative
        # matrix X.
        # Initialization Methods: 'random', 'nndsvd', 'nndsvda', 'nndsvdar'
        elif self.algorithm_ == 'nmf':
            model = NMF(n_components=self.target_space_, init='nndsvd', random_state=self.seed_)

        elif self.algorithm_ == 'nmf-random':
            model = NMF(n_components=self.target_space_, init='random', random_state=self.seed_)

        # Kernel PCA (Kernel: Radial Basis Function - RBF)
        elif self.algorithm_ == 'kpca-rbf':
            model = KernelPCA(n_components=self.target_space_, kernel="rbf")

        # Autoencoder (AE class)
        elif self.algorithm_ == 'AE':
            model = AE(n_components=self.target_space_, n_inputs=train_data.shape[1], random_state=self.seed_)

        train_data_red = model.fit_transform(train_data, test_data)
        test_data_red = model.transform(test_data)
        self.reduction_time_ = time.time() - t0

        return train_data_red, test_data_red

    def get_time(self):
        return self.reduction_time_
