import sys
import re
import gensim
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import load_vectors_300
from constants import TOTAL_TWEETS, MECHANISM_ID, UNWANTED_CHARS, EMBEDDING_FILE_NAME, SANITIZED_DATA_DIR, MODELS_DIR, MEDIAN_TWEET_WORDS

def load_sources():
  test_data = pd.read_csv("sources/haha_mechanism_target_test.csv")
  return test_data

def sanitize_tweet(tweet):
  for char in UNWANTED_CHARS:
    tweet = tweet.replace(char, ' ')
  tweet = re.sub('@\w*', '', tweet) #remove user references
  tweet = re.sub('\$', ' ', tweet) #split prices chars
  tweet = tweet.split(" ")
  tweet = [token for token in tweet if token != ''] #remove unwanted spaces
  return tweet

def get_sanitized_data(test_data):
  tweets = []

  for idx, row in test_data.iterrows():
    sanitized_tweet = sanitize_tweet(test_data.loc[idx, "text"])
    tweets.append(sanitized_tweet)

  return tweets

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

def classify_tweets(tweets, model_name):
  model = load_model(model_name)
  reversed_dictionary = {value : key for (key, value) in MECHANISM_ID.items()}

  predictions = np.argmax(model.predict(tweets), axis=-1)
  predictions = predictions.tolist()
  predictions = list(map(lambda x: x + 1, predictions))
  predictions = list(map(lambda x: reversed_dictionary[x] , predictions))

  return predictions

if __name__ == "__main__":
  model_name = 'models/lstm_gru_sigmoid_sigmoid_45'

  print('Loading test data (CSV)...')
  test_data = load_sources()
  print('Test data loaded ' + u'\u2705')

  print('Sanitizing tweets...')
  tweets = get_sanitized_data(test_data)
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
  indexes_tweets = pad_sequences(indexes_tweets, maxlen=MEDIAN_TWEET_WORDS)
  print('Sequenced and padding successfull' + u'\u2705')

  predictions = classify_tweets(indexes_tweets, model_name)

  test_data['mechanism'] = predictions
  test_data.to_csv('classifier_prediction.csv')

