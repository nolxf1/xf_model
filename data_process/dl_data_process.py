from xf_model.data_process.common_data_process import common_data_process

from xf_model.common.uitls import SparseFeat,DenseFeat

from xf_model.data_process.dl_embedding import get_fixlen_feature_names

import numpy as np

import warnings
warnings.filterwarnings("ignore")


class dl_data_process(common_data_process):

    def __init__(self, config):
        super(dl_data_process,self).__init__(config)
    def deepfm_get_standard_input(self):
        #处理空值
        self.deal_nan()
        #catencoder
        self.categlory_encoder()
        #mix_max
        self.deal_min_max()
        sparse_fea = self.fea_config['sparse_fea_config']
        dense_fea = self.fea_config['dense_fea_config']
        fixlen_feature_columns = [SparseFeat(col, self.data[col].nunique()) for col in
                                  sparse_fea] + [DenseFeat(col, 1) for col in dense_fea]
        linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns
        fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
        #此处遗留var len特征未处理
        data_model_input = [self.data[name].values for name in fixlen_feature_names]

        return data_model_input, self.data['label'].values, linear_feature_columns, dnn_feature_columns
    def deepfm_get_standard_input_train_and_val(self):
        #处理空值
        self.deal_nan()
        #catencoder
        self.categlory_encoder()
        #mix_max
        self.deal_min_max()
        sparse_fea = self.fea_config['sparse_fea_config']
        dense_fea = self.fea_config['dense_fea_config']
        par = self.fea_config['par']
        #print(par)
        fixlen_feature_columns = [SparseFeat(col, self.data[col].nunique()) for col in
                                  sparse_fea] + [DenseFeat(col, 1) for col in dense_fea]
        linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns
        fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
        #此处遗留var len特征未处理
        #此处划分训练集 验证集
        train_data = self.data.loc[self.data['leads_dt']<par,:]
        test_data = self.data.loc[self.data['leads_dt']>=par,:]
        
        data_model_input_train = [train_data[name].values for name in fixlen_feature_names]
        data_model_input_test = [test_data[name].values for name in fixlen_feature_names]
        print(test_data['label'].sum())
        return data_model_input_train,data_model_input_test, train_data['label'].values,test_data['label'].values, linear_feature_columns, dnn_feature_columns
    

    def mtl_deepfm_get_standard_input(self):
        #处理空值
        self.deal_nan()
        #catencoder
        self.categlory_encoder()
        #mix_max
        self.deal_min_max()
        sparse_fea = self.fea_config['sparse_fea_config']
        dense_fea = self.fea_config['dense_fea_config']
        label_fea = self.fea_config['label_fea_config']
        fixlen_feature_columns = [SparseFeat(col, self.data[col].nunique()) for col in
                                  sparse_fea] + [DenseFeat(col, 1) for col in dense_fea]
        linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns
        fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
        #此处遗留var len特征未处理
        data_model_input = [self.data[name].values for name in fixlen_feature_names]

        return data_model_input, [self.data[label_fea[i]].values for i in range(len(label_fea))], linear_feature_columns, dnn_feature_columns
    def mtl_deepfm_get_standard_input_train_and_val(self):
        #处理空值
        self.deal_nan()
        #catencoder
        self.categlory_encoder()
        #mix_max
        self.deal_min_max()
        sparse_fea = self.fea_config['sparse_fea_config']
        dense_fea = self.fea_config['dense_fea_config']
        label_fea = self.fea_config['label_fea_config']
        par = self.fea_config['par']
        #print(par)
        fixlen_feature_columns = [SparseFeat(col, self.data[col].nunique()) for col in
                                  sparse_fea] + [DenseFeat(col, 1) for col in dense_fea]
        linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns
        fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
        #此处遗留var len特征未处理
        #此处划分训练集 验证集
        train_data = self.data.loc[self.data['leads_dt']<par,:]
        test_data = self.data.loc[self.data['leads_dt']>=par,:]
        
        data_model_input_train = [train_data[name].values for name in fixlen_feature_names]
        data_model_input_test = [test_data[name].values for name in fixlen_feature_names]

        return data_model_input_train,data_model_input_test,[train_data[label_fea[i]].values for i in range(len(label_fea))], [test_data[label_fea[i]].values for i in range(len(label_fea))], linear_feature_columns, dnn_feature_columns


    def mix_loss_get_standard_input(self):
        # 处理空值
        self.deal_nan()
        # catencoder
        self.categlory_encoder()
        # mix_max
        self.deal_min_max()
        for item in self.data.columns.tolist():
            print(self.data[item].count())
        sparse_fea = self.fea_config['sparse_fea_config']
        dense_fea = self.fea_config['dense_fea_config']
        label_fea = self.fea_config['label_fea_config']
        extra_fea = self.fea_config['extra_fea_config']
        fixlen_feature_columns = [SparseFeat(col, self.data[col].nunique()) for col in
                                  sparse_fea] + [DenseFeat(col, 1) for col in dense_fea]
        linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns
        fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
        # 此处遗留var len特征未处理
        data_model_input = [self.data[name].values for name in fixlen_feature_names]
        data_target = []
        data_temp = []
        for item in self.data[label_fea+extra_fea].values:
            data_temp.append(np.array(item))
        data_target.append(np.array(data_temp))

        return data_model_input, data_target, linear_feature_columns, dnn_feature_columns
    
    def mix_loss_get_standard_input_train_and_val(self):
        # 处理空值
        self.deal_nan()
        # catencoder
        self.categlory_encoder()
        # mix_max
        self.deal_min_max()
        for item in self.data.columns.tolist():
            print(self.data[item].count())
        sparse_fea = self.fea_config['sparse_fea_config']
        dense_fea = self.fea_config['dense_fea_config']
        label_fea = self.fea_config['label_fea_config']
        extra_fea = self.fea_config['extra_fea_config']
        par = self.fea_config['par']
        fixlen_feature_columns = [SparseFeat(col, self.data[col].nunique()) for col in
                                  sparse_fea] + [DenseFeat(col, 1) for col in dense_fea]
        linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns
        fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
        # 此处遗留var len特征未处理
        
        train_data = self.data.loc[self.data.leads_dt<par,:]
        test_data = self.data.loc[self.data.leads_dt>=par,:]        
        
        data_model_input_train = [train_data[name].values for name in fixlen_feature_names]
        data_model_input_test = [test_data[name].values for name in fixlen_feature_names]
        data_target_train = []
        data_temp_train = []
        for item in train_data[label_fea+extra_fea].values:
            data_temp_train.append(np.array(item))
        data_target_train.append(np.array(data_temp_train))
        
        data_target_test = []
        data_temp_test = []
        for item in test_data[label_fea+extra_fea].values:
            data_temp_test.append(np.array(item))
        data_target_test.append(np.array(data_temp_test))
        

        return data_model_input_train,data_model_input_test,data_target_train, data_target_test, linear_feature_columns, dnn_feature_columns

    '''

    def multihot_encoder_for_train(data, sparse_features_multi_value):
        """
        :param data:
        :param sparse_features_multi_value:
        :return:
        """
        print('=' * 10, '> multihot_encoder_for_train')

        def toset(row):  # 去重,这样只能反应用户对uid的点击序列
            click_buid = ""
            for buid in set(row.split('|')):
                click_buid = click_buid + buid + "|"
            return click_buid[:-1]

        def split(x):
            key_ans = x.split('|')
            for key in key_ans:
                if key not in key2index:
                    # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for multi-hot input
                    key2index[key] = len(key2index) + 1
            return list(map(lambda x: key2index[x], key_ans))

        multihot_feature_lists = []
        multihot_maxlen_lists = []
        multihot_key2index_lists = []
        for col in tqdm(sparse_features_multi_value, desc='multihot_Encoder_for_Train'):
            trange(1, desc='multihot_encoder: ' + col, position=1, bar_format='{desc}')
            key2index = dict()
            if col == 'click_buid_list':
                data[col] = data[col].apply(toset)
            col_list = list(map(split, data[col].values))
            col_length = np.array(list(map(len, col_list)))
            # max_length = max(col_length)
            max_length = 49  # 将max_length写死(buid数量)
            col_list = pad_sequences(col_list, maxlen=max_length, padding='post', )
            multihot_feature_lists.append(col_list)
            multihot_maxlen_lists.append(max_length)
            multihot_key2index_lists.append(key2index)
        return multihot_feature_lists, multihot_maxlen_lists, multihot_key2index_lists

    def multihot_encoder_for_test(data, sparse_features_multi_value, multihot_key2index_lists):
        """
        :param data:
        :param sparse_features_multi_value:
        :param multihot_key2index_lists:
        :return:
        """
        print('=' * 10, '> multihot_encoder_for_test')
        count = 0

        def toset(row):  # 去重,这样只能反应用户对uid的点击序列
            click_buid = ""
            for buid in set(row.split('|')):
                click_buid = click_buid + buid + "|"
            return click_buid[:-1]

        def match(x):
            key_ans = x.split('|')
            for key in key_ans:
                if key not in multihot_key2index_lists[count]:
                    multihot_key2index_lists[count][key] = -1
            return list(map(lambda x: multihot_key2index_lists[count][x], key_ans))

        multihot_feature_lists = []
        for col in tqdm(sparse_features_multi_value, desc='multihot_Encoder_for_Test'):
            trange(1, desc='multihot_encoder: ' + col, position=1, bar_format='{desc}')
            if col == 'click_buid_list':
                data[col] = data[col].apply(toset)
            col_list = list(map(match, data[col].values))
            col_length = np.array(list(map(len, col_list)))
            # max_length = max(col_length)
            max_length = 49  # 将max_length写死(buid数量)
            col_list = pad_sequences(col_list, maxlen=max_length, padding='post', )
            multihot_feature_lists.append(col_list)
            count += 1
        return multihot_feature_lists
    '''
