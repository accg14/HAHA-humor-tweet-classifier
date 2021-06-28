import numpy as np
from numpy import loadtxt

from sklearn import model_selection
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier

from keras import utils

from constants import RANDOM_STATE_SEED, SANITIZED_DATA_DIR, TEST_SIZE, TARGET_CATEGORIES, EMBEDDING_MATRIX_DIR, FUNCTIONS, MEDIAN_TWEET_WORDS

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

def train_and_evaluate(model, X_train, X_test, Y_train, Y_test):
  my_callbacks = [
    EarlyStopping(monitor="loss", patience=PATIENTE, verbose=0, mode="auto", restore_best_weights= True)
  ]

  model.fit(x=X_train, y=Y_train, epochs=MAX_EPOCH, verbose=0, callbacks=my_callbacks)
  print("MODEL TRAINED " + u'\u2705')

  predictions = np.argmax(model.predict(X_test), axis=-1)
  Y_test_id = np.argmax(Y_test, axis=1)
  f1_obtained = f1_score(predictions, Y_test_id, average='macro')

  #print('F1 macro: ', f1_obtained)
  return model, f1_obtained

def classify_with_dummy_classifier(X_train, X_test, Y_train, Y_test):
  dummy_clf = DummyClassifier(strategy="stratified")
  dummy_clf.fit(X_train, Y_train)
  predictions = dummy_clf.predict(X_test)
  f1_obtained = f1_score(predictions, Y_test, average='macro')
  print("obtainded f1: "+ str(f1_obtained))



if __name__ == "__main__":
  embedding_matrix = get_embedding_matrix()
  print('EMBEDDING MATRIX LOADED ' + u'\u2705')
  X_train, X_test, Y_train, Y_test = get_splitted_data()
  print('DATA SPLITTED ' + u'\u2705')
  classify_with_dummy_classifier(X_train, X_test, Y_train, Y_test)
