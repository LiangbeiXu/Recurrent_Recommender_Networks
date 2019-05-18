# -*- Encoding:UTF-8 -*-

import tensorflow as tf
import numpy as np
import sys
from DataHelper import Data, AssistmentData
from tensorflow.keras.callbacks import TensorBoard

from keras.layers import Input, Embedding, Dot, Reshape, Dense, Add
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from keras import backend as K
from keras import regularizers, optimizers
from hyperopt import STATUS_OK, fmin, tpe, hp, Trials, space_eval
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time
import pickle

sys.path.insert(0, '/home/lxu/Documents/Probabilistic-Matrix-Factorization')
from  PreprocessAssistment import PreprocessAssistmentSkillBuilder, PreprocessAssistmentProblemSkill
from PreprocessAssistment import *
from IRT import IRT


def main():

    data_names = ['Assistment-09', 'Assistment-15']
    item_names = ['problem', 'skill']

    for data_name in data_names:
        for item in item_names:
            if not(data_name =='Assistment-15' and item == 'problem'):


                space = {
                    'embedding_size': hp.choice('embedding_size', [4, 8, 16, 32]),
                    'batch_size' : hp.choice('batch_size', [256, 512]),

                    'nb_epochs' :  hp.choice('nb_epochs', [8, 16, 32]),
                    'regularization': hp.choice('regularization',[ 0.0003,0.001, 0.003]),
                    'learning_rate': hp.choice('learning_rate',[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])

                }
                space = {
                    'embedding_size': hp.choice('embedding_size', [8]),
                    'batch_size' : hp.choice('batch_size', [256]),

                    'nb_epochs' :  hp.choice('nb_epochs', [16]),
                    'regularization': hp.choice('regularization',[ 0.0003]),
                    'learning_rate': hp.choice('learning_rate',[ 1e-3])

                }
                space = {
                    'embedding_size': [4],
                    'batch_size' : [256, 512],

                    'nb_epochs' :  [2],
                    'regularization': [  0.003],
                    'learning_rate': [  1e-3 ]
                }
                space = {
                    'embedding_size': [4, 8, 16, 32],
                    'batch_size' : [256, 512],
                    'nb_epochs' :  [8, 16, 32],
                    'regularization': [ 0.0003,0.001, 0.003],
                    'learning_rate': [ 1e-4, 1e-3, 2e-3]
                }

                print('Searching: '+ data_name + '_' + item)
                # run_a_trial(data_name, item, space)
                grid_search_models(data_name, item, space)

    # params = {'embedding_size':16, 'batch_size':512, 'nb_epochs': 16,'regularization':0.001, 'learning_rate': 1e-4}
    # data_name='Assistment-15'
    # item='skill'
    # ran_one_model(data_name, item, params)


def grid_search_models(data_name, item, search_space):
    keys, values = zip(*search_space.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print('Number of models: ', len(experiments))
    auc_val_all = []
    for exp in experiments:
        auc_train, auc_val = run_one_model(data_name,item, exp,'Searching')
        auc_val_all.append(auc_val)
    best_idx = auc_val_all.index(max(auc_val_all))
    print('Best model for '+ data_name + '_' + item)
    print(experiments[best_idx])
    auc_train, auc_val = run_one_model(data_name, item, experiments[best_idx], 'Testing')
    print('auc_train:', auc_train, 'auc_val', auc_val)


def run_one_model(data_name, item, params, mode):
    model = embedding(data_name, item, mode)
    auc_train, auc_val = model.train_evaluate(params)
    return auc_train, auc_val




def run_a_trial(data_name, item, space):
    max_evals = nb_evals = 1
    print("Attempt to resume a past training if it exists:")
    model = embedding(data_name, item, 'Searching')
    results_file_name = 'logs/results_'+ data_name + '_' + item + '.pkl'
    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open(results_file_name, "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        model.train_evaluate_wrapper,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open(results_file_name, "wb"))
    print("\nOPTIMIZATION STEP COMPLETE.\n")
    print("Best results yet (note that this is NOT calculated on the 'loss' "
          "metric despite the key is 'loss' - we rather take the negative "
          "best accuracy throughout learning as a metric to minimize):")
    print('Best results for '+ data_name + '_' + item)
    print(space_eval(space, best))
    auc_train, auc_val = run_one_model(data_name, item, space_eval(space, best), 'Testing')
    print('auc_train:', auc_train, 'auc_val', auc_val)


class embedding:
    def __init__(self, data_name='Assistment-09', item='problem', mode='Searching'):


        self.verbose = 2
        self.classification = True
        self.data_name = data_name
        self.item = item
        # get the data
        dataSet = AssistmentData(name=data_name, item=item)
        data = dataSet.data.values
        self.user_num = dataSet.user_num
        self.item_num = dataSet.item_num
        self.mode = mode
        if self.mode == 'Searching':
            self.train, self.test = train_test_split(data, test_size=0.2)
            self.train, self.validation = train_test_split(self.train, test_size = 0.1)
        elif self.mode == 'Testing':
            self.train, self.validation = train_test_split(data, test_size = 0.2)


    def train_evaluate_wrapper(self, params):
        auc_train, auc_val = self.train_evaluate(params)
        return {'loss':-auc_val, 'status': STATUS_OK}


    def train_evaluate(self, params):
        self.model_name = str(params) + '_' + str(int(time.time()))
        print('params test:', params)
        self.params = params
        self.embedding_model = self.generate_embedding_model()
        self.train_model()
        auc_train, auc_val = self.validate_model()
        return auc_train, auc_val


    def generate_embedding_model(self):
        """Model to embed users and wikiitems using the functional API.
            Trained to discern if a item is present in a article"""

        # Both inputs are 1-dimensional

        user = Input(name = 'user', shape = [1])
        item = Input(name = 'item', shape = [1])

        # Embedding the user (shape will be (None, 1, 50))
        user_embedding = Embedding(name = 'user_embedding',
                                   input_dim = (self.user_num),
                                   output_dim = self.params['embedding_size'],
                                   embeddings_regularizer=regularizers.l2(self.params['regularization']))(user)

        # Embedding the item (shape will be (None, 1, 50))
        item_embedding = Embedding(name = 'item_embedding',
                                   input_dim = (self.item_num),
                                   output_dim = self.params['embedding_size'],
                                   embeddings_regularizer=regularizers.l2(self.params['regularization']))(item)

        #  bias
        # global_bias = tf.keras.backend.variable(0, dtype=tf.float32, name='global_bias')
        user_bias = Embedding(name = 'user_bias',
                                   input_dim = (self.user_num),
                                   output_dim = 1,
                                   embeddings_regularizer=regularizers.l2(0.005))(user)
        item_bias = Embedding(name = 'item_bias',
                                   input_dim = (self.item_num),
                                   output_dim = 1,
                                   embeddings_regularizer=regularizers.l2(0.005))(item)

        # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
        merged = Dot(name = 'dot_product', normalize = True, axes = 2)([user_embedding, item_embedding])

        merged = Add()([merged, user_bias, item_bias])

        # Reshape to be a single number (shape will be (None, 1))
        merged = Reshape(target_shape = [1])(merged)


        # If classifcation, add extra layer and loss function is binary cross entropy

        # AUC for a binary classifier
        def auc(y_true, y_pred):
            ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
            pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
            pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
            binSizes = -(pfas[1:]-pfas[:-1])
            s = ptas*binSizes
            # return roc_auc_score(y_true, y_pred)
            return K.sum(s, axis=0)
        #-----------------------------------------------------------------------------------------------------------------------------------------------------
        # PFA, prob false alert for binary classifier
        def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
            y_pred = K.cast(y_pred >= threshold, 'float32')
            # N = total number of negative labels
            N = K.sum(1 - y_true)
            # FP = total number of false alerts, alerts from the negative class labels
            FP = K.sum(y_pred - y_pred * y_true)
            return FP/N
        #-----------------------------------------------------------------------------------------------------------------------------------------------------
        # P_TA prob true alerts for binary classifier
        def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
            y_pred = K.cast(y_pred >= threshold, 'float32')
            # P = total number of positive labels
            P = K.sum(y_true)
            # TP = total number of correct alerts, alerts from the positive class labels
            TP = K.sum(y_pred * y_true)
            return TP/P


        # opt_code = 'optimizers.' + self.params['optimizer']
        opt = optimizers.adam(lr=self.params['learning_rate'],beta_1=0.9, beta_2=0.999)
        if self.classification:
            merged = Dense(1, activation = 'sigmoid')(merged)
            model = Model(inputs = [user, item], outputs = merged)
            model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics=['accuracy', auc])

        # Otherwise loss function is mean squared error
        else:
            model = Model(inputs = [user, item], outputs = merged)
            model.compile(optimizer = self.params['optimizer'], loss = 'mse')

        return model


    def train_model(self):
        if 1:
            tensorboard = TensorBoard(log_dir="logs/{}".format(self.model_name))
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=50)
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
            logger = CSVLogger(filename="logs/{}".format(self.data_name + '-' + self.item + '' + self.model_name),append=False)
            h = self.embedding_model.fit(x=[self.train[:,0], self.train[:,1]],
                                         y = self.train[:,2],
                                         batch_size = self.params['batch_size'],
                                         epochs = self.params['nb_epochs'],
                                         # validation_split=0.2,
                                         validation_data = ([self.validation[:,0],self.validation[:,1]], self.validation[:,2]),
                                         verbose = self.verbose,
                                         callbacks=[es, mc, logger])
        # h = self.embedding_model.fit(x=[self.data[:,0], self.data[:,1]], y = self.data[:,2], validation_split=0.2,  epochs=self.epochs, batch_size=self.batch_size, shuffle=True,verbose =self.verbose)
    def validate_model(self):
        # saved_model = load_model('best_model.h5')
        pred_train =  self.embedding_model.predict(([self.train[:,0],self.train[:,1]]))
        pred_val =  self.embedding_model.predict(([self.validation[:,0],self.validation[:,1]]))
        auc_train = roc_auc_score(np.array(self.train[:,2], dtype=bool), pred_train)
        auc_val = roc_auc_score(np.array(self.validation[:,2], dtype=bool), pred_val)
        return auc_train, auc_val

if __name__ == '__main__':
    main()
