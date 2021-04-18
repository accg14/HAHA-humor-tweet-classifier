import sys
import numpy as np
from numpy import loadtxt

from sklearn import model_selection
from sklearn.metrics import accuracy_score, f1_score

import keras
from keras import layers
from keras import utils
from keras.callbacks import EarlyStopping

from constants import RANDOM_STATE_SEED, SANITIZED_DATA_DIR, TEST_SIZE, TARGET_CATEGORIES, EMBEDDING_MATRIX_DIR, PATIENTE, FUNCTIONS, RECURRENT_LAYER_UNITS, MAX_EPOCH, MEDIAN_TWEET_WORDS, MODELS_DIR

def get_embedding_matrix():
  return loadtxt(EMBEDDING_MATRIX_DIR+"embedding_matrix.csv", delimiter=',')

def get_splitted_data():
  inputs = loadtxt(SANITIZED_DATA_DIR+"tweets.csv", delimiter=',')
  target = loadtxt(SANITIZED_DATA_DIR+"mechanisms.csv", delimiter=',')

  X_train, X_test, Y_train, Y_test = model_selection.train_test_split(inputs, target, test_size = TEST_SIZE, random_state = RANDOM_STATE_SEED)

  Y_train = Y_train.tolist()
  Y_test = Y_test.tolist()

  Y_train = list(map(lambda x: x-1, Y_train))
  Y_test = list(map(lambda x: x-1, Y_test))

  Y_train = utils.to_categorical(Y_train, num_classes=TARGET_CATEGORIES)
  Y_test = utils.to_categorical(Y_test, num_classes=TARGET_CATEGORIES)

  Y_train = np.array(Y_train)
  Y_test = np.array(Y_test)

  return X_train, X_test, Y_train, Y_test

def compile_model(model_id, embedding_matrix, activation_function, recurrent_function, layer_units):
  input_dim = embedding_matrix.shape[0]
  output_dim = embedding_matrix.shape[1]

  if model_id == 0:
    model = keras.Sequential(
      [
        layers.Embedding(input_dim = input_dim, output_dim = output_dim, input_length = MEDIAN_TWEET_WORDS, weights = [embedding_matrix], trainable = False, mask_zero = True),
        layers.GRU(units = layer_units, activation = activation_function, return_sequences = True),
        layers.GRU(units = layer_units, activation = activation_function),
        layers.Dense(TARGET_CATEGORIES, activation = activation_function)
      ]
    )
  elif model_id == 1:
    model = keras.Sequential(
      [
        layers.Embedding(input_dim = input_dim, output_dim = output_dim, input_length = MEDIAN_TWEET_WORDS, weights = [embedding_matrix], trainable = False, mask_zero = True),
        layers.GRU(units = layer_units, activation = activation_function, return_sequences = True),
        layers.LSTM(units = layer_units, activation = activation_function),
        layers.Dense(TARGET_CATEGORIES, activation = activation_function)
      ]
    )
  elif model_id == 2:
    model = keras.Sequential(
      [
        layers.Embedding(input_dim = input_dim, output_dim = output_dim, input_length = MEDIAN_TWEET_WORDS, weights = [embedding_matrix], trainable = False, mask_zero = True),
        layers.LSTM(units = layer_units, activation = activation_function, return_sequences = True),
        layers.GRU(units = layer_units, activation = activation_function),
        layers.Dense(TARGET_CATEGORIES, activation = activation_function)
      ]
    )
  else:
    model = keras.Sequential(
      [
        layers.Embedding(input_dim = input_dim, output_dim = output_dim, input_length = MEDIAN_TWEET_WORDS, weights = [embedding_matrix], trainable = False, mask_zero = True),
        layers.LSTM(units = layer_units, activation = activation_function, return_sequences=True),
        layers.LSTM(units = layer_units, activation = activation_function),
        layers.Dense(TARGET_CATEGORIES, activation = activation_function)
      ]
    )
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  print("MODEL COMPILED " + u'\u2705')
  return model

def train_and_evaluate(model, X_train, X_test, Y_train, Y_test):
  my_callbacks = [
    EarlyStopping(monitor="loss", patience=PATIENTE, verbose=0, mode="auto", restore_best_weights= True)
  ]

  model.fit(x=X_train, y=Y_train, epochs=MAX_EPOCH, verbose=0, callbacks=my_callbacks)
  print("MODEL TRAINED " + u'\u2705')

  predictions = np.argmax(model.predict(X_test), axis=-1)
  Y_test_id = np.argmax(Y_test, axis=1)
  f1_obtained = f1_score(predictions, Y_test_id, average='macro')

  print('F1 macro: ', f1_obtained)
  return model, f1_obtained

def explore_models(model_id, model_name, embedding_matrix, X_train, X_test, Y_train, Y_test):
  best_model = { 'f1': 0}

  for function in FUNCTIONS:
    for layer_units in RECURRENT_LAYER_UNITS:
      print("--------------------------------------------")
      print("Model configuration:")
      print("  Activation function is: "+ function)
      print("  Recurrent function is: "+ function)
      print("  Layer units is: " + str(layer_units))

      model = compile_model(model_id, embedding_matrix, function, function, layer_units)
      model, f1_model = train_and_evaluate(model, X_train, X_test, Y_train, Y_test)

      if f1_model > best_model['f1']:
        best_model['f1'] = f1_model
        best_model['layer_units'] = layer_units
        best_model['function'] = function

      print('  f1: ' + str(f1_model))
      model_name = model_name + '_' + function + '_' + str(layer_units)
      model.save(MODELS_DIR+model_name)
      #del model
  print(best_model)

if __name__ == "__main__":
  model_id = int(sys.argv[1])
  model_name = sys.argv[2]

  embedding_matrix = get_embedding_matrix()
  print('EMBEDDING MATRIX LOADED ' + u'\u2705')
  X_train, X_test, Y_train, Y_test = get_splitted_data()
  print('DATA SPLITTED ' + u'\u2705')

  explore_models(model_id, model_name, embedding_matrix, X_train, X_test, Y_train, Y_test)
