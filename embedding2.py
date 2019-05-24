# -*- Encoding:UTF-8 -*-

import tensorflow as tf
# tf.enable_eager_execution()

import numpy as np
import sys
from DataHelper import Data, AssistmentData
from tensorflow.keras.callbacks import TensorBoard

from keras.layers import Input, Embedding, Dot, Reshape, Dense, Add, Activation, Lambda, RNN, GRU
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


from keras import backend as K
from keras import regularizers, optimizers
from hyperopt import STATUS_OK, fmin, tpe, hp, Trials, space_eval
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import time
import pickle
from multiprocessing import Pool
import pandas as pd

from pprint import pprint

sys.path.insert(0, '/home/lxu/Documents/Probabilistic-Matrix-Factorization')
from IRT import IRT


def main():
    if 0:
        data_names = ['Assistment15', 'Assistment09']
        # data_names = [ 'Assistment15']
        item_names = ['skill', 'problem']

        for data_name in data_names:
            for item in item_names:
                if not(data_name =='Assistment15' and item == 'problem'):


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
                        'embedding_size': [16, 32],
                        'batch_size' : [512],
                        'nb_epochs' :  [64],
                        'regularization': [ 0.0003,0.001, 0.003],
                        'learning_rate': [ 1e-4, 3e-4, 1e-3, 2e-3]
                    }
                    space = {
                        'embedding_size': hp.choice('embedding_size', [ 16, 32]),
                        'batch_size' : hp.choice('batch_size', [256, 512]),

                        'nb_epochs' :  hp.choice('nb_epochs', [32, 64]),
                        # 'regularization': hp.choice('regularization',[ 0.0003,0.001, 0.003]),
                        'regularization': hp.choice('regularization',[ 3e-4, 1e-3, 3e-3]),
                        'learning_rate': hp.choice('learning_rate', [1e-2, 3e-2, 1e-1, 3e-1, 1e0])

                    }

                    print('Searching: '+ data_name + '_' + item)
                    random_search_models(data_name, item, space)
                    random_search_models(data_name, item, space)
                    random_search_models(data_name, item, space)
                    # grid_search_models(data_name, item, space, parallel_mode=True)

    if 1:
        params = {'embedding_size':32, 'batch_size':512, 'nb_epochs': 5,'regularization':1e-4, 'learning_rate': 3}
        data_name='Assistment09'
        item='problem'
        em, auc_train, auc_test = run_one_model(data_name, item, params, 'Testing')
    if 0:
        params = {'embedding_size':32, 'batch_size':512, 'nb_epochs': 5,'regularization':1e-4, 'learning_rate': 3}
        data_name='Assistment09'
        item='problem'
        em, auc_train, auc_test = run_one_model(data_name, item, params, 'Testing')

        # create a factorization model

        fm = IRT(epsilon=4, _lambda=0.1, momentum=0.8, maxepoch=40, num_batches=300, batch_size=1000,\
                 problem=True, multi_skills=False, user_skill=False, user_prob=False, PFA=False, MF=True, \
                 num_feat=params['embedding_size'], MF_skill=False, user=True, skill_dyn_embeddding=False, skill=False, global_bias=False)

        train_data = pd.DataFrame(data=em.train[:,0:3], columns = ['user_id', 'problem_id', 'correct'])
        test_data = pd.DataFrame(data=em.validation[:,0:3], columns = ['user_id', 'problem_id', 'correct'])

        fm.fit(train_data, test_data, em.user_num, 5, em.item_num)

        # fm.beta_prob = item_bias.flatten()
        # fm.beta_user = user_bias.flatten()
        # fm.w_prob = item_embedding
        # fm.w_user = user_embedding


        validation_data = pd.DataFrame(data=em.validation[:,0:2], columns = ['user_id', 'problem_id'])
        y = em.validation[:,2]
        y_predicted = fm.predict(validation_data)
        print(em.validation[0:10,:])
        print(y_predicted[0:10])
        auc = roc_auc_score(np.array(y, dtype=bool), y_predicted)
        pred = y_predicted
        pred[y_predicted<0.5] = 0
        pred[y_predicted>=0.5] = 1

        acc = accuracy_score(np.array(y, dtype=bool), np.array(pred, dtype=bool))
        print('Test: auc %f, acc %f' %(auc, acc) )

        if 0:
            user_embedding = em.embedding_model.get_layer('user_embedding').get_weights()[0]
            item_embedding = em.embedding_model.get_layer('item_embedding').get_weights()[0]
            user_bias = em.embedding_model.get_layer('user_bias').get_weights()[0]
            item_bias = em.embedding_model.get_layer('item_bias').get_weights()[0]

            print('embeddings from NN:')
            print('item_bias', item_bias[0:10])
            print('user_bias', user_bias[0:10])
            print('item_embedding', item_embedding[0:10,:])
            print('user_embedding', user_embedding[0:10,:])

            print('embeddings from MF:')
            print('item_bias', fm.beta_prob[0:10])
            print('user_bias', fm.beta_user[0:10])
            print('item_embedding', fm.w_prob[0:10,:])
            print('user_embedding', fm.w_user[0:10,:])

        if 1:


            item_bias_ones = em.embedding_model.get_layer('item_bias').get_weights()
            for idx, ele in enumerate(item_bias_ones[0]):
                item_bias_ones[0][idx] = fm.beta_prob[idx]

            user_bias_ones = em.embedding_model.get_layer('user_bias').get_weights()
            for idx, ele in enumerate(user_bias_ones[0]):
                user_bias_ones[0][idx] = fm.beta_user[idx]

            item_embedding_ones = em.embedding_model.get_layer('item_embedding').get_weights()
            for idx, ele in enumerate(item_embedding_ones[0]):
                item_embedding_ones[0][idx] = fm.w_prob[idx,:]

            user_embedding_ones = em.embedding_model.get_layer('user_embedding').get_weights()
            for idx, ele in enumerate(user_embedding_ones[0]):
                user_embedding_ones[0][idx] = fm.w_user[idx,:]


            em.embedding_model.get_layer('item_bias').set_weights(item_bias_ones)
            em.embedding_model.get_layer('user_bias').set_weights(user_bias_ones)
            em.embedding_model.get_layer('item_embedding').set_weights(item_embedding_ones)
            em.embedding_model.get_layer('user_embedding').set_weights(user_embedding_ones)
            if 0 :
                print('New weights in NN:')

                user_embedding = em.embedding_model.get_layer('user_embedding').get_weights()[0]
                item_embedding = em.embedding_model.get_layer('item_embedding').get_weights()[0]
                user_bias = em.embedding_model.get_layer('user_bias').get_weights()[0]
                item_bias = em.embedding_model.get_layer('item_bias').get_weights()[0]

                print('embeddings from NN:')
                print('item_bias', item_bias[0:10])
                print('user_bias', user_bias[0:10])
                print('item_embedding', item_embedding[0:10,:])
                print('user_embedding', user_embedding[0:10,:])

            auc_train, auc_val = em.validate_model()
            print('Getting weights from MF. Test: auc_train %f, auc_val %f' %(auc_train, auc_val) )


            # free item embedding layer
            em.embedding_model.get_layer('item_embedding').trainable = False
            # small learning rate
            em.params['learning_rate'] = 1e-3
            em.params['nb_epochs'] = 30
            # use MF weights as initialization
            em.compile_model()

            if 0:
                user_embedding = em.embedding_model.get_layer('user_embedding').get_weights()[0]
                item_embedding = em.embedding_model.get_layer('item_embedding').get_weights()[0]
                user_bias = em.embedding_model.get_layer('user_bias').get_weights()[0]
                item_bias = em.embedding_model.get_layer('item_bias').get_weights()[0]
                print('embeddings from NN after complining:')
                print('item_bias', item_bias[0:10])
                print('user_bias', user_bias[0:10])
                print('item_embedding', item_embedding[0:10,:])
                print('user_embedding', user_embedding[0:10,:])

            em.train_model()

            if 0:
                user_embedding = em.embedding_model.get_layer('user_embedding').get_weights()[0]
                item_embedding = em.embedding_model.get_layer('item_embedding').get_weights()[0]
                user_bias = em.embedding_model.get_layer('user_bias').get_weights()[0]
                item_bias = em.embedding_model.get_layer('item_bias').get_weights()[0]
                print('embeddings from NN after retraining:')
                print('item_bias', item_bias[0:10])
                print('user_bias', user_bias[0:10])
                print('item_embedding', item_embedding[0:10,:])
                print('user_embedding', user_embedding[0:10,:])

            auc_train, auc_val = em.validate_model()
            print('Getting weights from MF and then train again. Test: auc_train %f, auc_val %f' %(auc_train, auc_val) )



def grid_search_models(data_name, item, search_space, parallel_mode):
    keys, values = zip(*search_space.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print('Number of models: ', len(experiments))

    args = [None] * np.size(data_name,0)
    if parallel_mode:
        for idx, exp in experiments:
            args[idx]  = [data_name, item, exp, 'Searching']
        pool = Pool(4)
        results = pool.map(run_one_model_wrapper, args)
        print(results)
    else:
        auc_val_all = []
        for exp in experiments:
            __, auc_train, auc_val = run_one_model(data_name,item, exp,'Searching')
            auc_val_all.append(auc_val)
        best_idx = auc_val_all.index(max(auc_val_all))
    print('Best model for '+ data_name + '_' + item)
    print(experiments[best_idx])
    __, auc_train, auc_val = run_one_model(data_name, item, experiments[best_idx], 'Testing')
    print('auc_train:', auc_train, 'auc_val', auc_val)


def run_one_model_wrapper(args):
    data_name, item, params, mode = args
    return run_one_model(data_name, item, params, mode)




def run_one_model(data_name, item, params, mode):
    model = embedding(data_name, item, mode)
    auc_train, auc_val = model.train_evaluate(params)
    return model, auc_train, auc_val




def random_search_models(data_name, item, space):
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
    __, auc_train, auc_val = run_one_model(data_name, item, space_eval(space, best), 'Testing')
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

        self.split_data(data=data, insample=True)

    def split_data(self, data, insample):
        if insample:
            if self.mode == 'Searching':
                self.train, self.test = train_test_split(data, test_size=0.2)
                self.train, self.validation = train_test_split(self.train, test_size = 0.1)
            elif self.mode == 'Testing':
                self.train, self.validation = train_test_split(data, test_size = 0.2)
        else:
            validation_size = 20000
            test_size = 60000
            if self.mode == 'Searching':
                self.test = data[-int(test_size):,:]
                self.train = data[0:-int(test_size+validation_size),:]
                self.validation = data[-int(test_size+validation_size):-int(test_size),:]
            elif self.mode == 'Testing':
                self.validation = data[-int(test_size):,:]
                self.train = data[0:-int(test_size),:]


    def train_evaluate_wrapper(self, params):
        auc_train, auc_val = self.train_evaluate(params)
        return {'loss':-auc_val, 'status': STATUS_OK}


    def train_evaluate(self, params):
        self.model_name = str(params) + '_' + str(int(time.time()))
        print('params test:', params)
        self.params = params
        self.embedding_model = self.generate_RRN_model()
        # self.embedding_model = self.generate_embedding_model()
        self.compile_model()
        self.train_model()
        auc_train, auc_val = self.validate_model()
        return auc_train, auc_val



    def generate_RRN_model(self):
        user = Input(name = 'user', shape=[1])
        item = Input(name = 'item', shape=[1])

        # one hot embedding
        def OneHot(input_dim=None, input_length=None):
            # Check if inputs were supplied correctly
            if input_dim is None or input_length is None:
                raise TypeError("input_dim or input_length is not set")

            # Helper method (not inlined for clarity)
            def _one_hot(x, num_classes):
                return K.one_hot(K.cast(x, 'uint8'),
                                  num_classes=num_classes)

            # Final layer representation as a Lambda layer
            return Lambda(_one_hot,
                          arguments={'num_classes': input_dim},
                          input_shape=(input_length,))

        #user_onehot = OneHot(input_dim=self.user_num)(user)
        # item_onehot = OneHot(input_dim=self.item_num)(item)
        if 0:
            user_output_shape = (None, self.user_num)
            user_onehot = Lambda(K.one_hot,
                              arguments={'nb_classes':self.user_num},
                              output_shape=user_output_shape)(user)

            item_output_shape = (None, self.item_num)
            item_onehot = Lambda(K.one_hot,
                                 arguments={'nb_classes':self.item_num},
                                 output_shape=item_output_shape)(item)

            user_embedding = Dense(name='user_embedding', units=self.params['embedding_size'], activation='relu')(user_onehot)
            item_embedding = Dense(name='item_embedding', units=self.params['embedding_size'], activation='relu')(item_onehot)


        user_embedding = Embedding(name = 'user_embedding',
                                   input_dim = (self.user_num),
                                   output_dim = self.params['embedding_size'],
                                   embeddings_regularizer=regularizers.l2(self.params['regularization']))(user)

        # Embedding the item (shape will be (None, 1, 50))
        item_embedding = Embedding(name = 'item_embedding',
                                   input_dim = (self.item_num),
                                   output_dim = self.params['embedding_size'],
                                   embeddings_regularizer=regularizers.l2(self.params['regularization']))(item)

        user_GRU = GRU(units=128, return_sequences=False)(item_embedding)
        item_GRU = GRU(units=128, return_sequences=False)(user_embedding)

        user_vec = Dense(units=64)(user_GRU)
        item_vec = Dense(units=64)(item_GRU)

        user_bias = Embedding(name='user_bias',
                                   input_dim=(self.user_num),
                                   output_dim=1,
                                   embeddings_regularizer=regularizers.l2(0.0))(user)
        item_bias = Embedding(name='item_bias',
                                   input_dim=(self.item_num),
                                   output_dim=1,
                                   embeddings_regularizer=regularizers.l2(0.0))(item)

        merged = Dot(name='dot_product', normalize=False, axes=1)([user_vec, item_vec])

        merged = Add()([merged, user_bias, item_bias])

        # If classification, add extra layer and loss function is binary cross entropy
        if self.classification:
            # merged = Dense(1, activation = 'sigmoid')(merged)
            merged = Activation('sigmoid')(merged)
            model = Model(inputs=[user, item], outputs=merged)
        # Otherwise loss function is mean squared error
        else:
            model = Model(inputs=[user, item], outputs=merged)

        return model


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
                                   embeddings_regularizer=regularizers.l2(0.0))(user)
        item_bias = Embedding(name = 'item_bias',
                                   input_dim = (self.item_num),
                                   output_dim = 1,
                                   embeddings_regularizer=regularizers.l2(0.0))(item)

        # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
        merged = Dot(name = 'dot_product', normalize = False, axes = 2)([user_embedding, item_embedding])

        merged = Add()([merged, user_bias, item_bias])

        # Reshape to be a single number (shape will be (None, 1))
        merged = Reshape(target_shape = [1])(merged)


        # If classifcation, add extra layer and loss function is binary cross entropy
        if self.classification:
            # merged = Dense(1, activation = 'sigmoid')(merged)
            merged = Activation('sigmoid')(merged)
            model = Model(inputs = [user, item], outputs = merged)
        # Otherwise loss function is mean squared error
        else:
            model = Model(inputs = [user, item], outputs = merged)

        return model

    def compile_model(self):
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

        opt = optimizers.SGD(lr=self.params['learning_rate'], decay=1e-6, momentum=0.9, nesterov=True)
        if self.classification:
            self.embedding_model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics=['accuracy', auc])

        # Otherwise loss function is mean squared error
        else:
            self.embedding_model.compile(optimizer = opt, loss = 'mse')


    def train_model(self):

        tensorboard = TensorBoard(log_dir="logs/{}".format(self.model_name))
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=100, min_delta=3e-4)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
        logger = CSVLogger(filename="logs/{}".format(self.data_name + '-' + self.item + '' + self.model_name),append=False)
        h = self.embedding_model.fit(x=[self.train[:,0], self.train[:,1]],
                                     y = self.train[:,2],
                                     batch_size = self.params['batch_size'],
                                     epochs = self.params['nb_epochs'],
                                     validation_data = ([self.validation[:,0],self.validation[:,1]], self.validation[:,2]),
                                     verbose = self.verbose,
                                     callbacks=[es])


    def validate_model(self):
        # saved_model = load_model('best_model.h5')
        pred_train = self.embedding_model.predict(([self.train[:,0],self.train[:,1]]))
        pred_val = self.embedding_model.predict(([self.validation[:,0],self.validation[:,1]]))

        print(self.validation[0:10,0:3])
        print(pred_val[0:10])
        auc_train = roc_auc_score(np.array(self.train[:,2], dtype=bool), pred_train)
        auc_val = roc_auc_score(np.array(self.validation[:,2], dtype=bool), pred_val)
        return auc_train, auc_val

if __name__ == '__main__':
    main()
