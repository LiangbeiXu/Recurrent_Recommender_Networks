# -*- Encoding:UTF-8 -*-

import tensorflow as tf
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from DataHelper import Data, AssistmentData
import pandas as pd
import matplotlib.pyplot as plt
import copy

sys.path.insert(0, '/home/lxu/Documents/Probabilistic-Matrix-Factorization')
from  PreprocessAssistment import PreprocessAssistmentSkillBuilder, PreprocessAssistmentProblemSkill
from PreprocessAssistment import *
from IRT import IRT
from IRT2 import IRT2




def main():

    model = RRN2(data_name='Assistment09', item='problem', epochs = 10)

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



class RRN2:

    def __init__(self, data_name='Assistment09', item='problem', epochs = 5):
        # params parser
        self.batch_size = 256
        self.n_step = 1
        self.lr = 1e-5
        self.verbose = 10
        self.item_embedding_dim = 10
        self.epochs = epochs
        self.train_loss = []
        self.train_loss_epoch = []
        self.val_loss = []
        self.val_loss_epoch = []
        self.train = []
        self.validation = []
        self.user_num = []
        self.item_num = []
        self.data_name = data_name
        self.item = item
        self.mode = 'Testing'
        self.prepare_data()
        self.calculate_embedding()
        print('number of users:',self.user_num, ', num of items:', self.item_num)
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


    def split_data(self, data, insample):
        if insample:
            if self.mode == 'Searching':
                self.train, self.test = train_test_split(data, test_size=0.2)
                self.train, self.validation = train_test_split(self.train, test_size = 0.1)
            elif self.mode == 'Testing':
                self.train, self.validation = train_test_split(data, test_size = 0.2)
        else:
            validation_size = 20000
            test_size = 20000
            if self.mode == 'Searching':
                self.test = data[-int(test_size):,:]
                self.train = data[0:-int(test_size+validation_size),:]
                self.validation = data[-int(test_size+validation_size):-int(test_size),:]
            elif self.mode == 'Testing':
                self.validation = data[-int(test_size):,:]
                self.train = data[0:-int(test_size),:]

        self.embedding_train, self.embedding_validation = train_test_split(self.train, test_size=0.2)
        self.train = pd.DataFrame(data=self.train[:,0:3], columns=['user_id', 'problem_id', 'correct'])
        self.validation = pd.DataFrame(data=self.validation[:,0:3], columns=['user_id', 'problem_id', 'correct'])
        self.embedding_train = pd.DataFrame(data=self.embedding_train[:,0:3], columns=['user_id', 'problem_id', 'correct'])
        self.embedding_validation = pd.DataFrame(data=self.embedding_validation[:,0:3], columns=['user_id', 'problem_id', 'correct'])
        if self.mode == 'Searching':
            self.test = pd.DataFrame(data=self.test[:,0:3], columns=['user_id', 'problem_id', 'correct'])

    def prepare_data(self):
        # read data
        dataSet = AssistmentData(name=self.data_name, item=self.item)
        data = dataSet.data.values
        self.user_num = dataSet.user_num
        self.item_num = dataSet.item_num
        # split data
        self.split_data(data=data, insample=False)




    def calculate_embedding(self):
        if 0:
            fm = IRT(epsilon=4, _lambda=0.1, momentum=0.8, maxepoch=200, num_batches=300, batch_size=1000,\
                     problem=True, multi_skills=False, user_skill=False, user_prob=False, PFA=False, MF=True,\
                     num_feat=self.item_embedding_dim, MF_skill=False, user=True, skill_dyn_embeddding=False, skill=False, global_bias=False)
            fm.fit(self.train, self.validation, self.user_num, 5, self.item_num)

        fm = IRT2(epsilon=5, _lambda=0.1, momentum=0.8, maxepoch=200, num_batches=300, batch_size=1000,
                problem=True, MF=True, num_feat=self.item_embedding_dim, user=True,  global_bias=True,
                problem_dyn_embedding=False)

        fm.fit(self.train, self.validation, self.user_num, self.item_num)


        # PlotLoss(model, name)
        print('global bias, user bias, skill bias', fm.beta_user, fm.beta_prob)
        self.user_bias_constant = fm.beta_user
        self.item_bias_constant = fm.beta_prob
        self.item_embedding_constant = fm.w_prob
        self.user_embedding_constant = fm.w_user



    def add_placeholder(self):
        # user placeholder
        self.userID = tf.placeholder(tf.int32, shape=[None, 1], name="userID")
        self.userID2 = tf.placeholder(tf.float32, shape=[None, self.item_embedding_dim], name="userID2")
        # movie placeholder
        self.movieID = tf.placeholder(tf.float32, shape=[None, self.item_embedding_dim], name="movieID")
        # target
        self.rating = tf.placeholder(tf.float32, shape=[None, 1], name="rating")
        # other params
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        self.user_bias = tf.placeholder(tf.float32, shape=[None, 1], name="user_bias")
        self.item_bias = tf.placeholder(tf.float32, shape=[None, 1], name="item_bias")


    def add_embedding_layer(self):
        with tf.name_scope("userID_embedding"):
            # user id embedding
            uid_onehot = tf.reshape(tf.one_hot(self.userID, self.user_num), shape=[-1, self.user_num])
            # uid_onehot_rating = tf.multiply(self.rating, uid_onehot)
            uid_layer = tf.layers.dense(uid_onehot, units=128, activation=tf.nn.relu, kernel_initializer=tf.initializers.zeros(), bias_initializer=tf.initializers.zeros())
            # self.uid_layer = tf.reshape(uid_layer, [-1, self.n_step, 128])
            self.uid_layer = tf.reshape(uid_layer, [-1, self.n_step, 128])

        with tf.name_scope("movie_embedding"):
            # movie id embedding
          #  mid_onehot = tf.reshape(tf.one_hot(self.movieID, self.item_num), shape=[-1, self.item_num])
            # mid_onehot_rating = tf.multiply(self.rating, mid_onehot)
            mid_layer = self.movieID
            self.mid_layer = tf.reshape(mid_layer, shape=[-1, self.n_step, self.item_embedding_dim])

    def add_rnn_layer(self):
        with tf.variable_scope("user_rnn_cell"):
            userCell = tf.nn.rnn_cell.GRUCell(num_units=128, kernel_initializer=tf.initializers.zeros, bias_initializer=tf.initializers.zeros())
            # userInput = self.movieID
            userInput = tf.transpose(self.mid_layer, [1, 0, 2])
            # movieInput = tf.transpose(tf.reshape(self.userID2, shape=[-1, self.n_step, 128]), [1, 0, 2])
            # cellInput = tf.concat([userInput, movieInput], 1)
            # userInput = tf.reshape(userInput, [-1, 128])
            # userInput = tf.split(userInput, self.n_step, axis=0)

            userOutputs, userStates = tf.nn.dynamic_rnn(userCell, userInput, dtype=tf.float32)
            # self.userOutput = userOutputs
            self.userOutput = userOutputs[-1]
        # with tf.variable_scope("movie_rnn_cell"):
            # movieCell = tf.nn.rnn_cell.GRUCell(num_units=128)

            # movieInput = tf.transpose(self.uid_layer, [1, 0, 2])
            # movieOutputs, movieStates = tf.nn.dynamic_rnn(movieCell, movieInput, dtype=tf.float32)
           #  self.movieOutput = movieOutputs[-1]

    def add_pred_layer(self):
        W = {
            'userOutput': tf.Variable(tf.random_normal(shape=[128, self.item_embedding_dim], stddev=0.001)),
            'movieOutput': tf.Variable(tf.random_normal(shape=[128, self.item_embedding_dim], stddev=0.001))
        }
        b = {
            'userOutput': tf.Variable(tf.random_normal(shape=[self.item_embedding_dim], stddev=0.001)),
            'movieOutput': tf.Variable(tf.random_normal(shape=[self.item_embedding_dim], stddev=0.001))
        }
        # userVector = tf.layers.dense(self.userOutput, units=self.item_embedding_dim, activation=tf.nn.relu, kernel_initializer=tf.initializers.zeros(), bias_initializer=tf.initializers.zeros())
        userVector = tf.add(tf.matmul(self.userOutput, W['userOutput']), b['userOutput'])
        # userVector2 = tf.add(self.userID2, userVector)
        # movieVector = tf.add(tf.matmul(self.movieOutput, W['movieOutput']), b['movieOutput'])
        # userVector = self.uid_layer
        movieVector  = self.movieID

        self.pred = tf.sigmoid(tf.reduce_sum(tf.multiply(movieVector, userVector ) + self.user_bias + self.item_bias, axis=1, keep_dims=True))

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
        userID2 = np.zeros(shape=(data.values.shape[0], self.item_embedding_dim), dtype=np.float32)
        movieID = np.zeros(shape=(data.values.shape[0], self.item_embedding_dim), dtype=np.float32)
        ratings = []

        user_bias = []
        item_bias = []
        # item_embedding = []
        idx = 0
        for i in data.values:
            userID.append([i[0]])
            userID2[idx, :] = (self.user_embedding_constant[np.int(i[0]), :])
            movieID[idx, :] = (self.item_embedding_constant[np.int(i[1]), :])
            ratings.append([float(i[2])])
            user_bias.append([copy.deepcopy(self.user_bias_constant[i[0]])])
            item_bias.append([copy.deepcopy(self.item_bias_constant[i[1]])])
            idx += 1

        return {
            self.userID: np.array(userID),
            self.userID2: userID2,
            self.movieID: movieID,
            self.rating: np.array(ratings),
            self.user_bias: np.array(user_bias),
            self.item_bias: np.array(item_bias),
            self.dropout: dropout
        }

    def predict(self):
        feed_dict = self.createFeedDict(self.validation)
        p = self.sess.run(self.pred, feed_dict=feed_dict)
        dt = pd.DataFrame({'act': self.validation.iloc[:,2], 'pred': p.reshape(-1)})
        auc = roc_auc_score(np.array(self.validation.iloc[:,2], dtype=bool), p.reshape(-1))
        return auc
        # print(dt[:5])
        # print("evaluation")
        # print("auc:", auc)

if __name__ == '__main__':
    main()
