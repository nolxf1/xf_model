
from xf_model.model.model_deepfm import PredictionLayer,DeepFM,DNN
import keras
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, mean_squared_error, mean_absolute_error
from xf_model.common.uitls import auc


class MTL_DeepFM:
    def __init__(self, model_config):
        self.model_config = model_config
        self.task1_net_units = self.model_config['task1_net_units']
        self.task2_net_units = self.model_config['task2_net_units']
        self.Optimizer_mtl = self.model_config['Optimizer_mtl']
        self.learning_rate_mtl = self.model_config['learning_rate_mtl']
        self.use_global_epochs = self.model_config['use_global_epochs']
        self.loss_weights = self.model_config['loss_weights']
        self.epochs = self.model_config['epochs']
        self.batch_size = self.model_config['batch_size']
        self.verbose = self.model_config['verbose']
        print('=' * 10, '> construct task1')
        deepfm1 = DeepFM(self.model_config['model1_config'])
        print('=' * 10, '> construct task2')
        deepfm2 = DeepFM(self.model_config['model2_config'])

        self.task1 = deepfm1
        self.task2 = deepfm2


    def model_(self):

        # 构建多任务模型
        print('=' * 10, '> construct soft mtl model')
        output1 = PredictionLayer(self.task1.task,name=self.task1.prefix)(self.task1.final_logit)
        output2 = PredictionLayer(self.task2.task,name=self.task2.prefix)(self.task2.final_logit)
        self.inputs_list = self.task1.inputs_list + self.task2.inputs_list
        model_mtl = keras.models.Model(inputs=self.inputs_list, outputs=[output1, output2])
        self.model_mtl =  model_mtl
        print(self.model_mtl.summary())

    def train(self,train_model_input,train_target):
        if self.Optimizer_mtl == 'Adam':
            optimizer_mtl = Adam(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'Adagrad':
            optimizer_mtl = Adagrad(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'RMSprop':
            optimizer_mtl = RMSprop(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'SGD':
            optimizer_mtl = SGD(lr=self.learning_rate_mtl)
        print('=' * 10, '> compile soft mtl')
        self.model_mtl.compile(optimizer=optimizer_mtl, loss=['binary_crossentropy', 'binary_crossentropy'], \
                          loss_weights=self.loss_weights,
                          metrics={self.task1.prefix:auc,self.task2.prefix:auc})
        if self.use_global_epochs:
            for i in range(self.epochs):
                print('=' * 10, '> global_epoch_{}'.format(i + 1))
                self.model_mtl.fit(
                    train_model_input,
                    train_target,
                    batch_size=self.batch_size,
                    epochs=1,
                    verbose=self.verbose,
                )
        else:
            self.model_mtl.fit(
                train_model_input,
                train_target,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                validation_data=[train_model_input, train_target]
            )
        pred_ans = self.model_mtl.predict(train_model_input, batch_size=2 ** 12)
        print('=' * 10, "> train " + 'task1' + " AUC: ", round(roc_auc_score(train_target[0], pred_ans[0]), 4))
        print('=' * 10, "> train " + 'task2' + " AUC: ", round(roc_auc_score(train_target[1], pred_ans[1]), 4))
    def train_(self,train_model_input,test_model_input,train_target,test_target):
        if self.Optimizer_mtl == 'Adam':
            optimizer_mtl = Adam(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'Adagrad':
            optimizer_mtl = Adagrad(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'RMSprop':
            optimizer_mtl = RMSprop(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'SGD':
            optimizer_mtl = SGD(lr=self.learning_rate_mtl)
        print('=' * 10, '> compile soft mtl')
        self.model_mtl.compile(optimizer=optimizer_mtl, loss=['binary_crossentropy', 'binary_crossentropy'], \
                          loss_weights=self.loss_weights,
                          metrics={self.task1.prefix:auc,self.task2.prefix:auc})
        if self.use_global_epochs:
            for i in range(self.epochs):
                print('=' * 10, '> global_epoch_{}'.format(i + 1))
                self.model_mtl.fit(
                    train_model_input,
                    train_target,
                    validation_data=(test_model_input,test_target),
                    batch_size=self.batch_size,
                    epochs=1,
                    verbose=self.verbose,
                )
        else:
            self.model_mtl.fit(
                train_model_input,
                train_target,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                validation_data=[train_model_input, train_target]
            )
        pred_ans = self.model_mtl.predict(train_model_input, batch_size=2 ** 12)
        pred_ans_ = self.model_mtl.predict(test_model_input, batch_size=2 ** 12)
        print('=' * 10, "> train " + 'task1' + " AUC: ", round(roc_auc_score(train_target[0], pred_ans[0]), 4))
        print('=' * 10, "> train " + 'task2' + " AUC: ", round(roc_auc_score(train_target[1], pred_ans[1]), 4))
        print('=' * 10, "> test " + 'task1' + " AUC: ", round(roc_auc_score(test_target[0], pred_ans[0]), 4))
        print('=' * 10, "> test " + 'task2' + " AUC: ", round(roc_auc_score(test_target[1], pred_ans[1]), 4))
class MTL_DeepFM_HARD:
    def __init__(self, model_config):
        self.model_config = model_config
        self.task1_net_units = self.model_config['task1_net_units']
        self.task2_net_units = self.model_config['task2_net_units']
        self.Optimizer_mtl = self.model_config['Optimizer_mtl']
        self.learning_rate_mtl = self.model_config['learning_rate_mtl']
        self.use_global_epochs = self.model_config['use_global_epochs']
        self.loss_weights = self.model_config['loss_weights']
        self.epochs = self.model_config['epochs']
        self.batch_size = self.model_config['batch_size']
        self.verbose = self.model_config['verbose']

        deepfm = DeepFM(self.model_config['model1_config'])
        
        self.deepfm = deepfm

        self.task1_output = PredictionLayer(deepfm.task,name='task1')(deepfm.final_logit)

        self.task2_output = PredictionLayer(deepfm.task,name='task2')(deepfm.final_logit)



    def model_(self):

        # 构建多任务模型
        print('=' * 10, '> construct hard mtl model')
        self.inputs_list = self.deepfm.inputs_list
        model_mtl = keras.models.Model(inputs=self.inputs_list, outputs=[self.task1_output, self.task2_output])
        self.model_mtl =  model_mtl
        print(self.model_mtl.summary())

    def train(self,train_model_input,train_target):
        if self.Optimizer_mtl == 'Adam':
            optimizer_mtl = Adam(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'Adagrad':
            optimizer_mtl = Adagrad(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'RMSprop':
            optimizer_mtl = RMSprop(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'SGD':
            optimizer_mtl = SGD(lr=self.learning_rate_mtl)
        print('=' * 10, '> compile hard mtl')
        self.model_mtl.compile(optimizer=optimizer_mtl, loss=['binary_crossentropy', 'binary_crossentropy'], \
                          loss_weights=self.loss_weights,
                          metrics={'task1':auc,'task2':auc})
        if self.use_global_epochs:
            for i in range(self.epochs):
                print('=' * 10, '> global_epoch_{}'.format(i + 1))
                self.model_mtl.fit(
                    train_model_input,
                    train_target,
                    batch_size=self.batch_size,
                    epochs=1,
                    verbose=self.verbose,
                )
        else:
            self.model_mtl.fit(
                train_model_input,
                train_target,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                validation_data=[train_model_input, train_target]
            )
        pred_ans = self.model_mtl.predict(train_model_input, batch_size=2 ** 12)
        print('=' * 10, "> train " + 'task1' + " AUC: ", round(roc_auc_score(train_target[0], pred_ans[0]), 4))
        print('=' * 10, "> train " + 'task2' + " AUC: ", round(roc_auc_score(train_target[1], pred_ans[1]), 4))

    def train_(self,train_model_input,test_model_input,train_target,test_target):
        if self.Optimizer_mtl == 'Adam':
            optimizer_mtl = Adam(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'Adagrad':
            optimizer_mtl = Adagrad(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'RMSprop':
            optimizer_mtl = RMSprop(lr=self.learning_rate_mtl)
        elif self.Optimizer_mtl == 'SGD':
            optimizer_mtl = SGD(lr=self.learning_rate_mtl)
        print('=' * 10, '> compile soft mtl')
        self.model_mtl.compile(optimizer=optimizer_mtl, loss=['binary_crossentropy', 'binary_crossentropy'], \
                          loss_weights=self.loss_weights,
                          metrics={self.task1.prefix:auc,self.task2.prefix:auc})
        if self.use_global_epochs:
            for i in range(self.epochs):
                print('=' * 10, '> global_epoch_{}'.format(i + 1))
                self.model_mtl.fit(
                    train_model_input,
                    train_target,
                    validation_data=(test_model_input,test_target),
                    batch_size=self.batch_size,
                    epochs=1,
                    verbose=self.verbose,
                )
        else:
            self.model_mtl.fit(
                train_model_input,
                train_target,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose,
                validation_data=[train_model_input, train_target]
            )
        pred_ans = self.model_mtl.predict(train_model_input, batch_size=2 ** 12)
        print('=' * 10, "> train " + 'task1' + " AUC: ", round(roc_auc_score(train_target[0], pred_ans[0]), 4))
        print('=' * 10, "> train " + 'task2' + " AUC: ", round(roc_auc_score(train_target[1], pred_ans[1]), 4))




