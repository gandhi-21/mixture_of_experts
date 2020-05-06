import numpy as np
import tensorflow as tf
import pandas as pd
import os
import warnings
from helper import conv_layer, pool, flatten_layer, fc_layer
from data_iterator import Data_Iterator
import pre_process
from sklearn.model_selection import train_test_split


class CNN():

    tf_sess = None
    model = None
    dataset = None
    batch_size = 128
    learning_rate = 0.001

    def __init__(self, X, y, data_iterator, enable_session=False, num_epochs=100, learning_rate=0.001, shape=None, num_classes=None):

        self.num_classes = num_classes
        self.data_iterator = data_iterator 
        self.x = X
        self.y = y
        if enable_session:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.6
            self.tf_sess = tf.Session(config=config)

        if data_iterator:
            self.data_iterator = data_iterator

        self.build_model(epochs=num_epochs,
                        learning_rate=learning_rate,
                        shape=shape,
                        num_classes=num_classes)

        
    def build_model(self, epochs=50, learning_rate=0.001, shape=None, num_classes=None):
        cross_entropy, correct = None, None

#        self.x = tf.reshape(self.x, [-1, 28, 28, 1])

        reshaped = tf.reshape(self.x, [-1, 28, 28, 1])

        # First layer
        c1_channels = 1
        c1_filters = 6
        c1 = conv_layer(input=reshaped, input_channels=c1_channels, filters=c1_filters, filters_size=5)
        # Pooling
        pool1 = pool(layer=c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        # Second layer
        c2_channels = 6
        c2_filters = 16
        c2 = conv_layer(input=pool1, input_channels=c2_channels, filters=c2_filters, filters_size=5)
        pool2 = pool(layer=c2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        # Flattened layer
        flattened = flatten_layer(layer=pool2)

        # First fully connected layer
        fc1_input = 784
        fc1_output = 120
        fc1 = fc_layer(input=flattened, inputs=fc1_input, outputs=fc1_output, relu=True)

        # Second fully connected layer
        fc2_input = 120
        fc2_output = 84
        fc2 = fc_layer(input=fc1, inputs=fc2_input, outputs=fc2_output, relu=True)

        # Logits
        l_inp = 84
        l_out = self.num_classes

        self.logits = fc_layer(input=fc2, inputs=l_inp, outputs=l_out, relu=False)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)

        self.loss = tf.reduce_mean(cross_entropy)

        correct = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.y, axis=1))

        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        self.prediction = tf.argmax(self.logits, axis=1)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


    def train(self, epochs, limit=6):
        print("Training model")

        print("shape of the input data is ", self.data_iterator.x_train.shape[0], self.data_iterator.y_train.shape)

        self.tf_sess.run(tf.global_variables_initializer())

        best, no_change, total_loss, total_acc = 0, 0, 0, 0

        for epoch in range(epochs):
            self.data_iterator.reset()
            try:
                total = 0
                while self.data_iterator.current_index < self.data_iterator.x_train.shape[0]:
                    batch_x, batch_y = self.data_iterator.next_batch()
                    
                    self.tf_sess.run(self.optimizer, feed_dict={self.x:batch_x, self.y:batch_y})
                    loss, acc = self.tf_sess.run([self.loss, self.accuracy], feed_dict={self.x:batch_x, self.y:batch_y})

                    total += acc * len(batch_y)
                    total_loss += loss * len(batch_y)
                    total_acc += acc * len(batch_y)

                    if self.data_iterator.current_index == self.data_iterator.x_train.shape[0]:
                        break

            except IndexError as error:
                pass
            print(f'done training epoch {epoch + 1}')
            validation_x, validation_y = self.data_iterator.get_validation()
            vloss, vacc = self.tf_sess.run([self.loss, self.accuracy], feed_dict={self.x:validation_x, self.y:validation_y})
            print(f'epoch {epoch + 1}: loss = {vloss:.4f}\n'
                  f'training accuracy = {total / len(self.data_iterator.y_train):.4f}\n'
                  f'validation accuracy = {vacc:.4f}\n',
                  f'learning_rate = {self.learning_rate:.10f}\n')
            
            # Early stopping
            if vacc > best:
                best = vacc
                no_change = 0
            else:
                no_change += 1

            if no_change >= limit:
                print('early stopping')
                break

        test_x, test_y = self.data_iterator.get_test()
        acc = self.tf_sess.run(self.accuracy, feed_dict={self.x: test_x, self.y: test_y})
        print(f'test accuracy = {acc:.4f}\n')

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Get the data
    
    print(os.getcwd())

    train_data = pd.read_csv("input /mnist_train.csv")
    test_data = pd.read_csv("input /mnist_test.csv")

    y_train = train_data.iloc[:,0]
    x_train = train_data.drop(train_data.columns[0], axis=1)

    y_test = test_data.iloc[:,0]
    x_test = test_data.drop(test_data.columns[0], axis=1)

    print("Shape of the data is ", x_train.shape, y_train.shape)

    train_x, valid_x, train_y, valid_y = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

    # Get the data iterator
    data_loader = Data_Iterator(train_x=train_x,
                                train_y=train_y,
                                valid_x=valid_x,
                                valid_y=valid_y,
                                test_x=x_test,
                                test_y=y_test,
                                batch_size=128)

    n_train = len(train_x)
    n_valid = len(valid_x)
    n_test = len(x_test)
    shape = 10
    n_classes = 10
    epochs=50
    learning_rate=1e-3

    X = tf.placeholder(tf.float32, shape=[None, 28*28])
    y = tf.placeholder(tf.float32, shape=[None, 10])


    cnn = CNN(X=X,
              y=y,
              data_iterator=data_loader,
              enable_session=True,
              num_epochs=epochs,
              learning_rate=learning_rate,
              shape=shape,
              num_classes=n_classes)

    # cnn.build_model(epochs=epochs,
    #                 learning_rate=learning_rate,
    #                 shape=shape,
    #                 num_classes=n_classes)
    cnn.train(epochs=epochs, limit=8)
    cnn.tf_sess.close()
    print("Done")