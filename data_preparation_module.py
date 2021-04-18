import re
import statistics
import gensim
import pandas as pd
import numpy as np
from numpy import savetxt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import load_vectors_300
from constants import TOTAL_TWEETS, MECHANISM_ID, UNWANTED_CHARS, EMBEDDING_FILE_NAME, SANITIZED_DATA_DIR, EMBEDDING_MATRIX_DIR

def load_sources():
  train_data = pd.read_csv("sources/haha_mechanism_target_train.csv")
  train_data.drop("id", axis=1, inplace=True)
  train_data.drop("target", axis=1, inplace=True)
  return train_data

def sanitize_tweet(tweet):
  for char in UNWANTED_CHARS:
    tweet = tweet.replace(char, ' ')
  tweet = re.sub('@\w*', '', tweet) #remove user references
  tweet = re.sub('\$', ' ', tweet) #split prices chars
  tweet = tweet.split(" ")
  tweet = [token for token in tweet if token != ''] #remove unwanted spaces
  return tweet

def get_sanitized_data(train_data):
  tweets = []
  mechanisms = []

  for idx, row in train_data.iterrows():
    mechanisms.append(MECHANISM_ID.get(train_data.loc[idx, "mechanism"]))

    sanitized_tweet = sanitize_tweet(train_data.loc[idx, "text"])
    tweets.append(sanitized_tweet)

  num_words_list = list(map(lambda x: len(x), tweets))
  median_words_tweet = statistics.median(num_words_list)
  print("The median words per tweet is: ", median_words_tweet)

  return tweets, mechanisms, median_words_tweet

def get_vector_word(embedding, word):
  if word in embedding:
    return embedding[word]
  elif word.capitalize() in embedding:
    return embedding[word.capitalize()]
  elif word.lower() in embedding:
    return embedding[word.lower()]
  elif word.upper() in embedding:
    return embedding[word.upper()]
  else:
    return None

def generate_embedding_matrix(embedding):
  embedding_words = list(embedding.vocab.keys())
  embedding_word_counter = len(embedding_words)
  tokenizer = Tokenizer(num_words=embedding_word_counter, filters='',
                                                    lower=False, split=' ', char_level=False, oov_token='<UNK>')
  tokenizer.fit_on_texts(embedding_words)
  embeddings_word_index = tokenizer.word_index
  embedding_matrix = np.zeros((embedding_word_counter + 2, 300))
  for word, index in embeddings_word_index.items():
      if index != 1:
          embedding_vector = get_vector_word(embedding, word)
          embedding_matrix[index] = embedding_vector
  return embedding_matrix, tokenizer

def persist_in_file(tweets, mechanisms, embedding_matrix):
  savetxt(SANITIZED_DATA_DIR + "tweets.csv", tweets, delimiter=',')
  mechanisms = np.array(mechanisms)
  savetxt(SANITIZED_DATA_DIR + "mechanisms.csv", mechanisms, delimiter=',')
  import pdb; pdb.set_trace()
  savetxt(EMBEDDING_MATRIX_DIR + "embedding_matrix.csv", embedding_matrix, delimiter=',')

def preprocess_data():
  print('Loading training data (CSV)...')
  train_data = load_sources()
  print('Train data loaded ' + u'\u2705')

  print('Sanitizing tweets, mechanisms and computing median words...')
  tweets, mechanisms, median_words_tweet = get_sanitized_data(train_data)
  print('Done' + u'\u2705')

  print('Importing embeddings from INCO file...')
  embedding = load_vectors_300.load(EMBEDDING_FILE_NAME)
  print('Embeddings imported ' + u'\u2705')
  print('Generating embedding matrix and related tokenizer...')
  embedding = embedding.wv
  embedding_matrix, tokenizer = generate_embedding_matrix(embedding)
  print('Embedding matrix generated successfully (tokenizer as well) ' + u'\u2705')

  print("Sequencing and padding tweets...")
  indexes_tweets = tokenizer.texts_to_sequences(tweets)
  indexes_tweets = pad_sequences(indexes_tweets, maxlen=median_words_tweet)
  print('Sequenced and padding successfull' + u'\u2705')

  print('Persisting sanitized data and embedding matrix...')
  persist_in_file(indexes_tweets, mechanisms, embedding_matrix)
  print('Sanitized data persisted successfully (embedding matrix as well)' + u'\u2705')
