import tensorflow as tf
import pandas as pd
import numpy as np

class data_iterator():

    def __init__(self, x_train, y_train, x_test, y_test, batch_size):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.batch_size = batch_size
        self.current_index = 0

    def next_batch(self):

        x_batch = self.x_train[self.current_index: self.current_index + self.batch_size]
        y_batch = self.y_train[self.current_index: self.current_index + self.batch_size]

        self.current_index += self.batch_size

        return x_batch, y_batch

    