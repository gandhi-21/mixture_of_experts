import tensorflow as tf
import pandas as pd
import numpy as np
from pre_process import normalize_data, shuffle_data, one_hot_encode


class Data_Iterator():

    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, test_y, batch_size):
        self.x_train = train_x
        self.x_test = test_x
        self.y_train = train_y
        self.y_test = test_y
        self.batch_size = batch_size
        self.current_index = 0
        self.validation_x = valid_x
        self.validation_y = valid_y
        self.num_classes = 10

        # Pre Process and normalize over here
        self.x_train = normalize_data(self.x_train)
        # Pre Process and one hot encode
        self.y_train = one_hot_encode(self.y_train)
        self.x_train, self.y_train = shuffle_data(self.x_train, self.y_train)

        # Pre Process the validation data here
        self.validation_x = normalize_data(self.validation_x)
        self.validation_y = one_hot_encode(self.validation_y)
        self.validation_x, self.validation_y = shuffle_data(self.validation_x, self.validation_y)

        # Pre process the test data here
        self.x_test = normalize_data(self.x_test)
        self.y_test = one_hot_encode(self.y_test)

        # self.x_train = tf.dtypes.cast(self.x_train, dtype=tf.float32)
        # self.x_test = tf.dtypes.cast(self.x_test, dtype=tf.float32)
        # self.validation_x = tf.dtypes.cast(self.validation_x, dtype=tf.float32)


    def next_batch(self):

        x_batch = self.x_train[self.current_index: self.current_index + self.batch_size]
        y_batch = self.y_train[self.current_index: self.current_index + self.batch_size]

        self.current_index += self.batch_size

        return x_batch, y_batch

    def get_validation(self):
        return self.validation_x, self.validation_y

    def get_test(self):
        return self.x_test, self.y_test

    def reset(self):
        self.current_index = 0