from abc import ABC
import os.path
from sklearn.metrics import classification_report
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.metrics import f1_score
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import Counter
import string
import os

'''
Collect the data set in a dict by tag to sentence
'''


def open_text_dict():
    file = open("dialog_acts.dat", "r")
    collect = dict()
    for line in file.readlines():
        sorting = line.lower().split(" ", 1)
        collect.setdefault(sorting[0], []).append(sorting[1])
    return collect


'''
Collect the data set in a list by item 1 the tag and item 2 the
sentence, X is the sentences and the Y the tags. 
'''


def open_text_list():
    file = open("dialog_acts.dat", "r")
    X = []
    Y = []
    collect = []
    for line in file.readlines():
        line = line[:-1]
        sorting = line.lower().split(" ", 1)
        collect.append(sorting)
        X.append(sorting[1])
        Y.append(sorting[0])
    return collect, X, Y


class classifier(ABC):
    # An abstract class to make sure each classifier has at least test_model and predict
    def test_model(self):
        pass

    def predict(self, sentence):
        pass


class base_model(classifier):
    # The most common classifier
    def __init__(self):
        self.data, self.X, self.Y = open_text_list()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, test_size=0.15)

    def count_tag(self, p):
        count = Counter()
        for y in self.Y:
            count[y] += 1
        return count.most_common(1)[0][0]

    def test_model(self):
        most_common = self.predict(self.X_test)
        print("Test baseline")
        print(classification_report(most_common, self.y_test))

    def predict(self, sentence):
        y_predict = []
        for p in sentence:
            y_predict.append(self.count_tag(p))
        return y_predict


class keyword(classifier):
    # The Keyword matching classifier
    def __init__(self):
        self.data, self.X, self.Y = open_text_list()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, test_size=0.15)

    def keyword_matching(self, sentence):
        current_score = -1
        current_tag = "Not Matched"
        for tag, sentence_learn in zip(self.y_train, self.X_train):
            word_match = 0
            for word in sentence.split(" "):
                if word in sentence_learn.split(" "):
                    word_match += 1
            if word_match / len(sentence_learn.split(" ")) > current_score:
                current_score = word_match / len(sentence_learn.split(" "))
                current_tag = tag
        return current_tag

    def test_model(self):
        predictions = self.predict(self.X_test)
        print("Test Keyword")
        print(classification_report(predictions, self.y_test))

    def predict(self, sentence):
        y_predict = []
        for s in sentence:
            y_predict.append(self.keyword_matching(s))
        return y_predict


class decision_tree(classifier):
    # The Decision Tree classifier

    def __init__(self):
        self.data, self.X, self.Y = open_text_list()
        self.count_vect = CountVectorizer()
        self.count_vect.fit(self.X)
        self.X = self.count_vect.transform(self.X)
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.Y)
        self.Y = self.le.transform(self.Y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, test_size=0.15)
        self.decTree = tree.DecisionTreeClassifier()
        self.decTree.fit(self.X_train, self.y_train)

    def test_model(self):
        print("Test Decision Tree")
        y_predict = self.decTree.predict(self.X_test)
        print(classification_report(self.le.inverse_transform(
            self.y_test), self.le.inverse_transform(y_predict)))

    def predict(self, sentence):
        return self.le.inverse_transform(self.decTree.predict(self.count_vect.transform(sentence)))


class logistic_regression(classifier):
    # The logistic regression classifier

    def __init__(self):
        self.data, self.X, self.Y = open_text_list()
        self.count_vect = CountVectorizer()
        self.count_vect.fit(self.X)
        self.X = self.count_vect.transform(self.X)
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.Y)
        self.Y = self.le.transform(self.Y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, test_size=0.15)
        self.logistic = LogisticRegression().fit(self.X_train, self.y_train)

    def test_model(self):
        print("Test Logistic")
        y_predict = self.logistic.predict(self.X_test)
        print(classification_report(self.le.inverse_transform(
            self.y_test), self.le.inverse_transform(y_predict)))

    def predict(self, sentence):
        return self.le.inverse_transform(self.logistic.predict(self.count_vect.transform(sentence)))


class nn(classifier):
    # The LSTM classifier

    def __init__(self):
        tf.get_logger().setLevel('ERROR')
        self.data, self.X, self.Y = open_text_list()
        test_data = [x.split(" ") for x in self.X]
        unique_words = len(
            set([item for sublist in test_data for item in sublist]))
        self.max_words = unique_words + 1
        self.tokenizer = Tokenizer(
            num_words=self.max_words,
            filters='"#$%&()*+-/:;<=>@[\]^_`{|}~'
        )
        self.tokenizer.fit_on_texts(self.X)
        self.max_phrase_len = max(len(elem) for elem in test_data)
        self.X = self.tokenizer.texts_to_sequences(self.X)
        self.X = pad_sequences(self.X, maxlen=self.max_phrase_len)
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.Y)
        self.Y = self.le.transform(self.Y)
        self.Y = to_categorical(self.Y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, test_size=0.15)
        current_dir = os.getcwd()
        model_dir = os.path.join(current_dir, "LTSM_model")
        if os.path.exists(model_dir):
            self.model_lstm = keras.models.load_model(model_dir)
        else:
            self.model_lstm = Sequential()
            self.model_lstm.add(Embedding(
                input_dim=self.max_words, output_dim=256, input_length=self.max_phrase_len))
            self.model_lstm.add(SpatialDropout1D(0.3))
            self.model_lstm.add(LSTM(256, dropout=0.3, recurrent_dropout=0.3))
            self.model_lstm.add(Dense(256, activation='relu'))
            self.model_lstm.add(Dropout(0.3))
            self.model_lstm.add(Dense(15, activation='softmax'))
            self.model_lstm.compile(
                loss='categorical_crossentropy',
                optimizer='Adam',
                metrics=['accuracy']
            )
            self.history = self.model_lstm.fit(
                self.X_train,
                self.y_train,
                validation_split=0.2,
                epochs=10,
                batch_size=128
            )
            self.model_lstm.save(model_dir)

    def predict(self, sentence):
        sentence = self.tokenizer.texts_to_sequences([sentence[0]])
        sentence = pad_sequences(sentence, maxlen=self.max_phrase_len)
        pre = self.model_lstm.predict(sentence)
        pre = np.argmax(pre, axis=1)
        return self.le.inverse_transform(pre)

    def test_model(self):
        res = self.model_lstm.predict(self.X_test, batch_size=64, verbose=1)
        y_pred_bool = np.argmax(res, axis=1)
        y_t = np.argmax(self.y_test, axis=1)
        print("Test NN")
        print(classification_report(self.le.inverse_transform(
            y_t), self.le.inverse_transform(y_pred_bool)))
