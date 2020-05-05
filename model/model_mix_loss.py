

from xf_model.model.model_deepfm import PredictionLayer,DeepFM,DNN
import keras
from keras.callbacks import EarlyStopping
from keras import Model
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
from xf_model.common.uitls import auc,mixed_loss,concat_fun,mixed_auc
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, mean_squared_error, mean_absolute_error
import numpy as np
import tensorflow as tf
class MIX_LOSS:
    def __init__(self, model_config):
        self.model_config = model_config
        self.task1_net_units = self.model_config['task1_net_units']
        self.Optimizer_mtl = self.model_config['Optimizer_mtl']
        self.learning_rate_mtl = self.model_config['learning_rate_mtl']
        self.use_global_epochs = self.model_config['use_global_epochs']
        self.epochs = self.model_config['epochs']
        self.batch_size = self.model_config['batch_size']
        self.verbose = self.model_config['verbose']
        
        
        
        deepfm = DeepFM(self.model_config['model1_config'])
        print('=' * 10, '> construct exp linear')
        target2_logit = keras.layers.Dense(1, use_bias=False, activation=tf.exp,name='exp')(deepfm.dnn_input)
        output_target1 = PredictionLayer(deepfm.task,name=deepfm.prefix)(deepfm.final_logit)
        output_target2 = target2_logit
        outputs = concat_fun([output_target1,output_target2])
        self.input_list_ = deepfm.inputs_list
        print('=' * 10, '> construct mix loss model')
        model_mix = keras.models.Model(inputs=deepfm.inputs_list, outputs=outputs)
        print(model_mix.summary())
        self.model_mix =  model_mix

    def train(self,train_model_input,train_target):
        if self.Optimizer_mtl == 'Adam':
            optimizer_mtl = Adam(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'Adagrad':
            optimizer_mtl = Adagrad(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'RMSprop':
            optimizer_mtl = RMSprop(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'SGD':
            optimizer_mtl = SGD(lr=self.learning_rate_mtl)
        print('=' * 10, '> compile mix loss')
        self.model_mix.compile(optimizer=optimizer_mtl, loss=mixed_loss)
        if self.use_global_epochs:
            for i in range(self.epochs):
                print('=' * 10, '> global_epoch_{}'.format(i + 1))
                self.model_mix.fit(
                    train_model_input,
                    train_target,
                    batch_size=self.batch_size,
                    epochs=1,
                    verbose=self.verbose,
                )
        else:
            my_callbacks = [EarlyStopping(monitor=mixed_loss, patience=20, verbose=1, mode='max')]
            self.model_mix.fit(
                train_model_input,
                train_target,
                batch_size=self.batch_size,
                validation_data=(test_model_input,test_target),
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=my_callbacks
            )



        pred_ans = self.model_mix.predict(train_model_input, batch_size=2 ** 12)
        #print(pred_ans)
        print('=' * 10, "> train " + 'deepfm' + " AUC: ", round(roc_auc_score(np.array(train_target)[:,:,0].reshape(-1), pred_ans[:,0]), 4))
        
        #print('pred_ans')
        #print(pred_ans)
        '''
        tf.cast(pred_ans, tf.float32)

        with tf.Session() as sess:
            print(sess.run(
                mixed_loss(np.array(train_target[0]), pred_ans)))
        print(np.array(train_target)[:,:,0].reshape(-1))

        print(pred_ans[:,0])


        print('=' * 10, "> train " + 'task1' + " AUC: ", round(auc(np.array(train_target)[:,:,1].reshape(-1), pred_ans[:,0]), 4))
        '''
    def train_(self,train_model_input,test_model_input,train_target,test_target):
        if self.Optimizer_mtl == 'Adam':
            optimizer_mtl = Adam(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'Adagrad':
            optimizer_mtl = Adagrad(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'RMSprop':
            optimizer_mtl = RMSprop(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'SGD':
            optimizer_mtl = SGD(lr=self.learning_rate_mtl)
        print('=' * 10, '> compile mix loss')
        self.model_mix.compile(optimizer=optimizer_mtl, loss=mixed_loss,metrics=[mixed_loss])
        if self.use_global_epochs:
            for i in range(self.epochs):
                print('=' * 10, '> global_epoch_{}'.format(i + 1))
                self.model_mix.fit(
                    train_model_input,
                    train_target,
                    validation_data=(test_model_input,test_target),
                    batch_size=self.batch_size,
                    epochs=1,
                    verbose=self.verbose,
                )
                pred_ans = self.model_mix.predict(test_model_input, batch_size=2 ** 12)
                #print(pred_ans)
                print('=' * 10, "> test " + 'deepfm' + " AUC: ", round(roc_auc_score(np.array(test_target)[:,:,0].reshape(-1),                     pred_ans[:,0]), 4))
                

        else:
            my_callbacks = [EarlyStopping(monitor='val_mixed_loss', patience=3, verbose=1, mode='min')]
            self.model_mix.fit(
                train_model_input,
                train_target,
                batch_size=self.batch_size,
                validation_data=(test_model_input,test_target),
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=my_callbacks
            )



        pred_ans = self.model_mix.predict(train_model_input, batch_size=2 ** 12)
        #print(pred_ans)
        print('=' * 10, "> train " + 'deepfm' + " AUC: ", round(roc_auc_score(np.array(train_target)[:,:,0].reshape(-1), pred_ans[:,0]), 4))
        pred_ans = self.model_mix.predict(test_model_input, batch_size=2 ** 12)
        #print(pred_ans)
        print('=' * 10, "> test " + 'deepfm' + " AUC: ", round(roc_auc_score(np.array(test_target)[:,:,0].reshape(-1), pred_ans[:,0]), 4))










