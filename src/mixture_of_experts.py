import numpy as np
import tensorflow as tf
import pandas as pd
import os
import warnings
from helper import conv_layer, pool, flatten_layer, fc_layer
from data_iterator import Data_Iterator
import pre_process
from sklearn.model_selection import train_test_split
from model5 import CNN as MNISTBase
import tensorflow.contrib.slim as slim

class mixture_of_experts:

    tf_sess = None
    model = None
    batch_size = 128
    learning_rate = 0.001
    num_mixtures = None
    num_inputs = 400
    networks = list()

    def __init__(self, data_iterator, enable_session=False, num_mixtures=4, num_epochs=100, learning_rate=0.001,
                shape=None, num_classes=None):

        if enable_session:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.6
            self.tf_sess = tf.Session(config=config)

        self.num_mixtures = num_mixtures
        self.data_iterator = data_iterator
        self.num_inputs = self.data_iterator.x_train.shape[1]
        self.learning_rate = learning_rate
        self.shape = shape
        self.num_classes = num_classes
        self.build_model(epochs=num_epochs,
                        learning_rate=learning_rate,
                        shape=shape,
                        num_classes=num_classes)

    
    def build_model(self, epochs=50, learning_rate=0.001, shape=None, num_classes=None):

        with tf.variable_scope('moe', reuse=True) as scope:
            self.x = tf.placeholder(tf.float32, [None, self.num_inputs], name='x')
            self.y = tf.placeholder(tf.float32, [None, self.num_classes], name='y')

        # lenet = MNISTBase(X=self.x, y=self.y, data_iterator=self.data_iterator, num_epochs=epochs,
        #                  learning_rate==learning_rate=,
        #                  shape=shape,
        #                 num_classes=num_classes)

        for x in range(self.num_mixtures):
            self.networks.append(
                MNISTBase(X=self.x, y=self.y, data_iterator=self.data_iterator, enable_session=True, num_epochs=epochs,
                          learning_rate=learning_rate, shape=self.shape, num_classes=self.num_classes)
            )
        
        print("Checkpoint")

        concat = tf.concat([expert.logits for expert in self.networks], axis=1)

        gate_activations = slim.fully_connected(
            self.x, # figure out the input dimensions
            self.num_classes * (self.num_mixtures + 1),
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(1e-8),
            scope="gates"
        )

        expert_activations = slim.fully_connected(
            self.x,
            self.num_classes * self.num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(1e-8),
            scope="experts"
        )

        gating_distributions = tf.nn.softmax(
            tf.reshape(
                gate_activations,
                [-1, self.num_mixtures + 1]))
        
        expert_distribution = tf.nn.softmax(
            tf.reshape(
                expert_activations,
                [-1, self.num_mixtures]))

        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distributions[:,:self.num_mixtures] * expert_distribution, 1)
        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                        [-1, self.num_classes])
                    
        
        self.logits = tf.reshape(final_probabilities, [-1, self.num_classes])

        for expert in self.networks:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=expert.logits, labels=self.y)
            expert.loss = tf.reduce_mean(cross_entropy)
            expert.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(expert.loss)
            correct = tf.equal(tf.argmax(expert.logits, axis=1), tf.argmax(self.y, axis=1))
            expert.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
        self.loss = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        correct = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.y, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        self.predict = tf.nn.softmax(self.logits)

    
    def train_model(self, epochs, limit=4):
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

                    for index, expert in enumerate(self.networks):
                        self.tf_sess.run([expert.optimizer], feed_dict={self.x: batch_x, self.y: batch_y})


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
        
        # Evaluate the accuracy for each expert
        for index, expert in enumerate(self.networks):
            loss, acc = self.tf_sess.run([expert.loss, expert.accuracy], feed_dict={self.x: test_x, self.y: test_y})
            print(f'\tExpert {index + 1}:\ntest loss = {loss:.4f}\ntest_accuracy={acc:.4f}')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
    epochs = 10
    learning_rate = 1e-3

    moe = mixture_of_experts(enable_session=True, data_iterator=data_loader,
                             num_mixtures=4, num_epochs=epochs, learning_rate=learning_rate,
                             shape=shape, num_classes=n_classes)
    moe.train_model(epochs=epochs, limit=5)
    moe.tf_sess.close()
