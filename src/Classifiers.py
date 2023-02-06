import time

# Supervised classifiers.
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from LSTM import LSTMClassifier, BiLSTMClassifier

# Evaluation libraries
from sklearn.metrics import classification_report


class TextClassifiers:
    def __init__(self, classes, latent_dim, seed=42):
        self.models_ = {
            "Logistic Regression": [
                LogisticRegression(max_iter=300, random_state=seed),
                0.0, 0.0],
            "Decision Tree": [
                DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=None, random_state=seed),
                0.0, 0.0],
            "Random Forest": [
                RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, max_features='sqrt',
                                       n_jobs=8, random_state=seed),
                0.0, 0.0],
            "SVM (RBF Kernel)": [SVC(kernel='rbf', C=1, random_state=seed), 0.0, 0.0],
            "FCFF Neural Net:":
                [MLPClassifier(activation='relu', hidden_layer_sizes=(64, 16), random_state=seed), 0.0, 0.0],
            "LSTM": [
                LSTMClassifier(classes, latent_dim, seed),
                0.0, 0.0],
            "BiDirectional LSTM": [
                BiLSTMClassifier(classes, latent_dim, seed),
                0.0, 0.0]
        }

    def train(self, x_train, y_train):
        for mdl in self.models_:
            print("\tTraining", mdl + "...", end="", flush=True)
            t0 = time.time()

            clf = self.models_[mdl][0]
            clf.fit(x_train, y_train)

            print(" completed in %5.3f sec." % (time.time() - t0))

    def test(self, x_test, y_test):
        for mdl in self.models_:
            print("\tTesting", mdl + "...", end="", flush=True)
            t0 = time.time()

            clf = self.models_[mdl][0]
            y_predicted = clf.predict(x_test)

            print(" completed in %5.2f sec." % (time.time() - t0))

            print(classification_report(y_test, y_predicted, digits=4))

            if hasattr(clf, 'eval') and callable(clf.eval):
                clf.eval(x_test, y_test)
