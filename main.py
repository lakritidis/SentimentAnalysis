from Classifiers import TextClassifiers
from Datasets import TextDataset

RANDOM_SEED = 42

DATASET_PATH = 'C:/Users/Leo/Documents/JupyterNotebooks/Springer CSSN/datasets/'

DATASETS = [
    # {'location': 'movie_data.csv', 'text': 'review', 'class': 'sentiment'},
    # {'location': 'Twitter_Financial_Sentiment.csv', 'text': 'cleaned_tweets', 'class': 'sentiment'},
    # {'location': 'Twitter_US_Airline_Sentiment.csv', 'text': 'text', 'class': 'airline_sentiment'},
    # {'location': 'sentiment140.csv', 'text': 5, 'class': 0},
    {'location': 'Industrial_and_Scientific_5.json.gz', 'text': 'reviewText', 'class': 'overall'},
    ]

LATENT_DIM = 500

VECTORIZERS = ['tf-idf', 'tf-idf-dmf', 'word2vec']
# REDUCERS = [None, 'tsvd', 'nmf', 'kpca-rbf', 'AE', 'CVAE', 'CAE', 'autoencoder']
REDUCERS = [None, 'tsvd', 'nmf', 'kpca-rbf', 'AE', 'fs-best']

d = 0
v = 0
r = 0
if v == 2:
    r = 0

ds = TextDataset(DATASET_PATH + DATASETS[d]['location'], DATASETS[d]['text'], DATASETS[d]['class'],
                 VECTORIZERS[v], RANDOM_SEED)

print(80 * '=')
print("=== Text Vectorizer:", VECTORIZERS[v])
if r == 0:
    print("=== Running on the original feature space (without dimensionality reduction)")
else:
    print("=== Reduce to", LATENT_DIM, "dimensions with", REDUCERS[r])

x_train, x_test, y_train, y_test = ds.process(
    vectorizer=VECTORIZERS[v], reduction_method=REDUCERS[r], latent_dim=LATENT_DIM)

if r == 0:
    LATENT_DIM = ds.dimensionality_

cls = TextClassifiers(ds.classes_, LATENT_DIM, RANDOM_SEED)
cls.train(x_train, y_train)
cls.test(x_test, y_test)
