from xf_model.model.model_dl_base import PredictionLayer,FM,DNN
from xf_model.data_process.dl_embedding import build_input_features,combined_dnn_input_dense_embed,\
    embedding_input_from_feature_columns,get_dense_input,get_linear_logit
from keras.models import Model
from keras.callbacks import EarlyStopping
from xf_model.common.uitls import auc,concat_fun
from sklearn.metrics import roc_auc_score
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
import keras
class DeepFM:
    def __init__(self,model_config):

        #初始化模型参数
        self.linear_feature_columns = model_config['linear_feature_columns']
        self.dnn_feature_columns = model_config['dnn_feature_columns']
        self.embedding_size = model_config['embedding_size']
        self.use_fm = model_config['use_fm']
        self.dnn_hidden_units = model_config['dnn_hidden_units']
        self.l2_reg_linear = model_config['l2_reg_linear']
        self.l2_reg_embedding = model_config['l2_reg_embedding']
        self.l2_reg_dnn = model_config['l2_reg_dnn']
        self.init_std = model_config['init_std']
        self.seed = model_config['seed']
        self.dnn_dropout = model_config['dnn_dropout']
        self.dnn_activation = model_config['dnn_activation']
        self.dnn_use_bn = model_config['dnn_use_bn']
        self.Optimizer = model_config['Optimizer']
        self.learning_rate = model_config['learning_rate']
        self.use_global_epochs = model_config['use_global_epochs']
        self.epochs = model_config['epochs']
        self.verbose = model_config['verbose']
        self.batch_size =  model_config['batch_size']
        self.prefix =  model_config['prefix']
        self.task = model_config['task']

        print('=' * 10, '> construct DeepFM')

        #模型输入特征
        self.features = build_input_features(self.linear_feature_columns + self.dnn_feature_columns,prefix=self.prefix)

        #模型输入向量
        self.inputs_list = list(self.features.values())

        #embedding
        sparse_embedding_list, dense_embedding_list = embedding_input_from_feature_columns(self.features, self.dnn_feature_columns,
                                                                            self.embedding_size, self.l2_reg_embedding, self.init_std,
                                                                             self.seed,prefix=self.prefix)
        self.sparse_embedding_list = sparse_embedding_list
        self.dense_embedding_list = dense_embedding_list

        #dense not embedding
        self.dense_value_list = get_dense_input(self.features,self.dnn_feature_columns)
        #线性部分
        linear_logit = get_linear_logit(self.features, self.linear_feature_columns, l2_reg=self.l2_reg_linear, init_std=self.init_std,
                                        seed=self.seed, prefix='linear'+self.prefix)
        #fm部分
        fm_input = concat_fun(self.sparse_embedding_list+self.dense_embedding_list, axis=1)
        fm_logit = FM()(fm_input)
        dnn_input = combined_dnn_input_dense_embed(self.sparse_embedding_list,self.dense_embedding_list)
        self.dnn_input = dnn_input
        dnn_out = DNN(self.dnn_hidden_units, self.dnn_activation, self.l2_reg_dnn, self.dnn_dropout, self.dnn_use_bn, self.seed)(dnn_input)
        dnn_logit = keras.layers.Dense(1, use_bias=False, activation=None)(dnn_out)

        if len(self.dnn_hidden_units) == 0 and self.use_fm == False:  # only linear
            final_logit = linear_logit
        elif len(self.dnn_hidden_units) == 0 and self.use_fm == True:  # linear + FM
            final_logit = keras.layers.add([linear_logit, fm_logit])
        elif len(self.dnn_hidden_units) > 0 and self.use_fm == False:  # linear + Deep
            final_logit = keras.layers.add([linear_logit, dnn_logit])
        elif len(self.dnn_hidden_units) > 0 and self.use_fm == True:  # linear + FM + Deep
            final_logit = keras.layers.add([linear_logit, fm_logit, dnn_logit])
        else:
            raise NotImplementedError

        self.final_logit = final_logit


        #output = PredictionLayer(self.task)(final_logit)
        #self.output = output


    def model_(self):
        self.output = PredictionLayer(self.task,name=self.prefix)(self.final_logit)
        model =  Model(inputs=self.inputs_list, outputs=self.output)
        self.model = model
        print(model.summary())

    def train(self,train_model_input,train_target):

        if self.Optimizer == 'Adam':
            optimizer = Adam(lr=self.learning_rate)
        elif self.Optimizer == 'Adagrad':
            optimizer = Adagrad(lr=self.learning_rate)
        elif self.Optimizer == 'RMSprop':
            optimizer = RMSprop(lr=self.learning_rate)
        elif self.Optimizer == 'SGD':
            optimizer = SGD(lr=self.learning_rate)
        print('=' * 10, '> compile DeepFM')
        self.model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['binary_crossentropy', auc])
        if self.use_global_epochs:
            for i in range(self.epochs):
                print('=' * 10, '> global_epoch_{}'.format(i + 1))
                self.model.fit(
                    train_model_input,
                    train_target,
                    batch_size=self.batch_size,
                    epochs=1,
                    verbose=self.verbose
                )
            '''
            # 取某一层的输出为输出新建为model，采用函数模型
            for item in self.model.layers:
                print(item.name)
                print(isinstance(item, keras.engine.input_layer.InputLayer))
                if isinstance(item, keras.engine.input_layer.InputLayer):
                    print("input_layer")
                else:
                    layer_model = Model(inputs=self.inputs_list,
                                       outputs=self.model.get_layer(item.name).output)
                    #以这个model的预测值作为输出
                    layer_output = layer_model.predict(train_model_input, batch_size=1024)
                    print(layer_output)
            '''
        else:
            self.model.fit(
                train_model_input,
                train_target,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose
            )

        pred_ans = self.model.predict(train_model_input, batch_size=2 ** 12)
        print('=' * 10, "> train AUC", round(roc_auc_score(train_target, pred_ans), 4))
        
    def train_(self,train_model_input,test_model_input,train_target,test_target):

        if self.Optimizer == 'Adam':
            optimizer = Adam(lr=self.learning_rate)
        elif self.Optimizer == 'Adagrad':
            optimizer = Adagrad(lr=self.learning_rate)
        elif self.Optimizer == 'RMSprop':
            optimizer = RMSprop(lr=self.learning_rate)
        elif self.Optimizer == 'SGD':
            optimizer = SGD(lr=self.learning_rate)
        print('=' * 10, '> compile DeepFM')
        self.model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['binary_crossentropy', auc])
        if self.use_global_epochs:
            for i in range(self.epochs):
                print('=' * 10, '> global_epoch_{}'.format(i + 1))
                self.model.fit(
                    train_model_input,
                    train_target,
                    validation_data=(train_model_input,train_target),
                    batch_size=self.batch_size,
                    epochs=1,
                    verbose=self.verbose
                )
                pred_ans = self.model.predict(test_model_input, batch_size=2 ** 12)
                print('=' * 10, "> test AUC", round(roc_auc_score(test_target, pred_ans), 4))
            '''
            # 取某一层的输出为输出新建为model，采用函数模型
            for item in self.model.layers:
                print(item.name)
                print(isinstance(item, keras.engine.input_layer.InputLayer))
                if isinstance(item, keras.engine.input_layer.InputLayer):
                    print("input_layer")
                else:
                    layer_model = Model(inputs=self.inputs_list,
                                       outputs=self.model.get_layer(item.name).output)
                    #以这个model的预测值作为输出
                    layer_output = layer_model.predict(train_model_input, batch_size=1024)
                    print(layer_output)
            '''
        else:
            my_callbacks = [EarlyStopping(monitor='val_binary_crossentropy', patience=5, verbose=1, mode='min')]
            self.model.fit(
                train_model_input,
                train_target,
                batch_size=self.batch_size,
                validation_data=(test_model_input,test_target),
                epochs=self.epochs,
                verbose=self.verbose,
                callbacks=my_callbacks
            )

        pred_ans = self.model.predict(train_model_input, batch_size=2 ** 12)
        pred_t_ans = self.model.predict(test_model_input, batch_size=2 ** 12)
        print('=' * 10, "> train AUC", round(roc_auc_score(train_target, pred_ans), 4))
        pred_ans = self.model.predict(train_model_input, batch_size=2 ** 12)
        print('=' * 10, "> test AUC", round(roc_auc_score(test_target, pred_t_ans), 4))

