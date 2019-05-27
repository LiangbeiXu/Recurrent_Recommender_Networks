# -*- Encoding:UTF-8 -*-

import tensorflow as tf
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from DataHelper import Data, AssistmentData
import pandas as pd
import matplotlib.pyplot as plt


def main():
    if 0:
        model = RRN(data_name='Assistment09', item='skill', epochs = 3)
        model.run()
        auc = model.predict()
        print("evaluation","auc:", auc)

        plt.plot(model.train_loss_epoch, model.train_loss, marker='*', label='loss: Train Data')
        plt.plot(model.val_loss_epoch, model.val_loss, marker='*', label='AUC: Test Data')
        plt.xlabel('Number of Batches')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
        plt.show()

        model = None
    if 1:
        model = RRN(data_name='Assistment09', item='problem', epochs = 5)
        model.run()
        auc = model.predict()
        print("evaluation","auc:", auc)
        plt.plot(model.train_loss_epoch, model.train_loss, marker='*', label='loss: Train Data')
        plt.plot(model.val_loss_epoch, model.val_loss, marker='*', label='AUC: Test Data')
        plt.xlabel('Number of Batches')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
        plt.show()
        # tf.reset_default_graph()
        model = None
    if 0:
        model = RRN(data_name='Assistment-15', item='skill', epochs = 3)
        model.run()
        auc = model.predict()
        print( "evaluation","auc:", auc)
        plt.plot(model.train_loss_epoch, model.train_loss, marker='*', label='loss: Train Data')
        plt.plot(model.val_loss_epoch, model.val_loss, marker='*', label='AUC: Test Data')
        plt.xlabel('Number of Batches')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
        plt.show()

class RRN:

    def __init__(self, data_name='Assistment-09', item='problem', epochs = 1):
        # params parser
        self.batch_size = 200
        self.n_step = 1
        self.lr = 5e-4
        self.verbose = 10
        self.epochs = epochs
        self.train_loss = []
        self.train_loss_epoch = []
        self.val_loss = []
        self.val_loss_epoch = []
        # Data
        dataSet = AssistmentData(name=data_name, item=item)
        data = dataSet.data.values
        #self.train, self.validation = train_test_split(data, test_size = 0.01)
        self.user_num = dataSet.user_num
        self.item_num = dataSet.item_num
        print('number of users:',self.user_num, ', num of items:', self.item_num)
        self.train = data[0:-20000,:]
        self.validation = data[-20000:,:]
        print('train size:', self.train.shape[0], 'validation size', self.validation.shape[0])
        # Model
        tf.reset_default_graph()
        self.add_placeholder()
        self.add_embedding_layer()
        self.add_rnn_layer()
        self.add_pred_layer()
        self.add_loss()
        self.add_train_step()
        self.init_session()



    def add_placeholder(self):
        # user placeholder
        self.userID = tf.placeholder(tf.int32, shape=[None, 1], name="userID")
        # movie placeholder
        self.movieID = tf.placeholder(tf.int32, shape=[None, 1], name="movieID")
        # target
        self.rating = tf.placeholder(tf.float32, shape=[None, 1], name="rating")
        # other params
        self.dropout = tf.placeholder(tf.float32, name='dropout')

    def add_embedding_layer(self):
        with tf.name_scope("userID_embedding"):
            # user id embedding
            uid_onehot = tf.reshape(tf.one_hot(self.userID, self.user_num), shape=[-1, self.user_num])
            # uid_onehot_rating = tf.multiply(self.rating, uid_onehot)
            uid_layer = tf.layers.dense(uid_onehot, units=128, activation=tf.nn.relu)
            self.uid_layer = tf.reshape(uid_layer, [-1, self.n_step, 128])

        with tf.name_scope("movie_embedding"):
            # movie id embedding
            mid_onehot = tf.reshape(tf.one_hot(self.movieID, self.item_num), shape=[-1, self.item_num])
            # mid_onehot_rating = tf.multiply(self.rating, mid_onehot)
            mid_layer = tf.layers.dense(mid_onehot, units=128, activation=tf.nn.relu)
            self.mid_layer = tf.reshape(mid_layer, shape=[-1, self.n_step, 128])

    def add_rnn_layer(self):
        with tf.variable_scope("user_rnn_cell"):
            userCell = tf.nn.rnn_cell.LSTMCell(num_units=128)
            # dropout_cell = DropoutWrapper(lstm_cell, input_keep_prob=self.keep_rate, output_keep_prob=self.keep_rate, state_keep_prob=self.keep_rate)

            userInput = tf.transpose(self.mid_layer, [1, 0, 2])
            # userInput = tf.reshape(userInput, [-1, 128])
            # userInput = tf.split(userInput, self.n_step, axis=0)

            userOutputs, userStates = tf.nn.dynamic_rnn(userCell, userInput, dtype=tf.float32)
            self.userOutput = userOutputs[-1]
        with tf.variable_scope("movie_rnn_cell"):
            movieCell = tf.nn.rnn_cell.LSTMCell(num_units=128)

            movieInput = tf.transpose(self.uid_layer, [1, 0, 2])
            movieOutputs, movieStates = tf.nn.dynamic_rnn(movieCell, movieInput, dtype=tf.float32)
            self.movieOutput = movieOutputs[-1]

    def add_pred_layer(self):
        W = {
            'userOutput': tf.Variable(tf.random_normal(shape=[128, 64], stddev=0.1)),
            'movieOutput': tf.Variable(tf.random_normal(shape=[128, 64], stddev=0.1))
        }
        b = {
            'userOutput': tf.Variable(tf.random_normal(shape=[64], stddev=0.1)),
            'movieOutput': tf.Variable(tf.random_normal(shape=[64], stddev=0.1))
        }
        userVector = tf.add(tf.matmul(self.userOutput, W['userOutput']), b['userOutput'])
        movieVector = tf.add(tf.matmul(self.movieOutput, W['movieOutput']), b['movieOutput'])

        self.pred = tf.sigmoid(tf.reduce_sum(tf.multiply(userVector, movieVector), axis=1, keep_dims=True))

    def add_loss(self):
        losses = tf.losses.log_loss(self.rating, self.pred)
        self.loss = tf.reduce_mean(losses)

    def add_train_step(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def init_session(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())

    def run(self):
        length = len(self.train)
        batches = length // self.batch_size + 1

        # shuffled_idx = np.random.permutation(np.arange(len(self.train)))
        # self.train = self.train[shuffled_idx]


        for ii in range(self.epochs):
            for i in range(batches):
                minIdx = i * self.batch_size
                maxIdx = min(length, (i+1)*self.batch_size)
                train_batch = self.train[minIdx:maxIdx]
                feed_dict = self.createFeedDict(train_batch)

                tmpLoss = self.sess.run(self.loss, feed_dict=feed_dict)
                self.train_loss.append(tmpLoss)
                self.train_loss_epoch.append((ii)*batches + i)
                self.sess.run(self.train_op, feed_dict=feed_dict)

                if self.verbose and i % (self.verbose * 10) == 0:
                    val_auc = self.predict()
                    self.val_loss_epoch.append((ii)*batches + i)
                    self.val_loss.append(val_auc)
                    sys.stdout.write('\r{} / {}： val auc = {}'.format(
                        i, batches, val_auc
                    ))
                    sys.stdout.flush()

                #if self.verbose and i % self.verbose == 0:
                #    sys.stdout.write('\r{} / {}： loss = {}'.format(
                #        i, batches, np.sqrt(np.mean(train_loss[-20:]))
                #    ))
                #    sys.stdout.flush()
        print("Training Finish, Last 2000 batches loss is {}.".format(
            np.sqrt(np.mean(self.train_loss[-2000:]))
        ))

    def createFeedDict(self, data, dropout=1.):
        userID = []
        movieID = []
        ratings = []
        for i in data:
            userID.append([i[0]-1])
            movieID.append([i[1]-1])
            ratings.append([float(i[2])])
        return {
            self.userID: np.array(userID),
            self.movieID: np.array(movieID),
            self.rating: np.array(ratings),
            self.dropout: dropout
        }

    def predict(self):
        feed_dict = self.createFeedDict(self.validation)
        p = self.sess.run(self.pred, feed_dict=feed_dict)
        dt = pd.DataFrame({'act': self.validation[:,2], 'pred': p.reshape(-1)})
        auc = roc_auc_score(np.array(self.validation[:,2], dtype=bool), p.reshape(-1))
        return auc
        # print(dt[:5])
        # print("evaluation")
        # print("auc:", auc)

if __name__ == '__main__':
    main()
