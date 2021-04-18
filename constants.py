TOTAL_TWEETS = 4887
MECHANISM_ID = {
        'absurd': 1,
        'analogy': 2,
        'embarrassment': 3,
        'exaggeration': 4,
        'insults': 5,
        'irony': 6,
        'misunderstanding': 7,
        'parody': 8,
        'reference': 9,
        'stereotype': 10,
        'unmasking': 11,
        'wordplay': 12
    }
TARGET_CATEGORIES = 12

FUNCTIONS = ['elu', 'relu', 'selu','sigmoid', 'tanh']
RECURRENT_LAYER_UNITS = [15, 30, 75]
MAX_EPOCH = 80

UNWANTED_CHARS = ['!', ',', '"', '-', '...','–','XD', 'xD', '¿', '?', '—', '\n', "#", '¡', ':', "“", '.', '(', ')',"¬¬", "\('.')/", "*", '\n', '»', '\x97', '\x85']

EMBEDDING_FILE_NAME = 'emb39-word2vec'
SANITIZED_DATA_DIR = 'sanitized_data/'
EMBEDDING_MATRIX_DIR = 'sources/'
MODELS_DIR = 'models/'

TEST_SIZE = 0.25
TOTAL_MODELS = 4
RANDOM_STATE_SEED=14

PATIENTE = 10

MEDIAN_TWEET_WORDS = 18
