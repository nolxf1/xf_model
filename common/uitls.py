#通用工具开发每个function只处理一个特征
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,StandardScaler
from collections import namedtuple
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from tqdm import tqdm, trange
from keras.engine.topology import Layer
import tensorflow as tf
from sklearn.metrics import roc_auc_score
'''
缺失值处理
'''
def fill_nan_by_mean(fea):
    fea.fillna(fea.mean(),inplace=True)
    return fea
def fill_nan_by_mode(fea):
    fea.fillna(fea.mode()[0],inplace=True)
    return fea
def fill_nan_by_mid(fea):
    fea.fillna(fea.quantile(0.5),inplace=True)
    return fea
def fill_nan_by_key(fea,key):
    fea.fillna(key,inplace=True)
    return fea
'''
异常值检测
'''
def check_specail_by_std_mean(fea):
    # 平均值+/-3*标准差”之外的数据标识为异常值
    item_mean = fea.mean()
    item_std = fea.std()
    item_low = item_mean - 3 * item_std
    item_up = item_mean + 3 * item_std
    fea.map(lambda x: np.nan if (x < item_low) or (x > item_up) else x)
    return fea
def check_specail_by_quantile(fea):
    # 4分位数异常值检测法
    q1 = np.percentile(np.array(fea), 25)
    q3 = np.percentile(np.array(fea), 75)
    ior = q3 - q1
    item_low = q1 - 1.5 * ior
    item_up = q3 + 1.5 * ior
    fea.map(lambda x: np.nan if (x < item_low) or (x > item_up) else x)
    return fea
'''
特征分箱
'''
def bin_featrue(fea,bins):
    def bins_(x):
        if x>=bins['left'] and x<bins['right']:
            return bins['key']
    fea = fea.map(bins_)
    return fea
'''
取对数
'''
def log_featrue(fea):
    fea = fea.map(lambda x: np.log1p(x))
    return fea
'''
归一化
'''
def min_max_feature(fea,dlimit=0,ulimit=1):
    mms = MinMaxScaler(feature_range=(dlimit, ulimit))
    fea = mms.fit_transform(fea.values.reshape(-1, 1))
    return fea
'''
标准化
'''
def standard_scaler_feature(fea):
    ss = StandardScaler()
    fea = ss.fit_transform(fea.values.reshape(-1,1))
    return fea
'''
one-hot 维度降低
'''
def drop_axis(fea,limit=0.95):
    temp_fea_dict = dict(fea.value_counts())
    all_len = len(fea)
    sum = 0
    temp_lis = []
    for k,v in temp_fea_dict.items():
        sum+=v
        temp_lis.append(k)
        if sum/all_len>limit:
            break
    print("save dim",temp_lis)
    fea[fea.isin(temp_lis)==False]='un'
    return fea
'''
one-hot
'''
def one_hot_list(Index, Num):
    one_hot = [0] * Num
    one_hot[Index] = 1
    return one_hot


def oneHot(fea, values,column):
    join_vec = []
    value_num = len(values)
    for item in fea:
        index = values.index(item)
        join_vec.append(one_hot_list(index, value_num))

    new_columns = [column + '_' + str(i) for i in values]
    one_hot = pd.DataFrame(join_vec, columns=new_columns)
    return one_hot

def oneHotEncoder(Data, oneHotCols, valueDict, Drop=True):
    print('=' * 10, '> one_hot data')
    for column in oneHotCols:
        join_vec = []
        valuelist = valueDict[column]
        value_num = len(valuelist)
        for item in Data[column]:
            index = valuelist.index(item)
            join_vec.append(one_hot_list(index, value_num))

        new_columns = [column + '_' + str(i) for i in valuelist]
        one_hot = pd.DataFrame(join_vec, columns=new_columns)
        Data = Data.join(one_hot)
        if Drop:
            Data.drop(column, axis=1, inplace=True)
            Data.reset_index(drop=True, inplace=True)
    return Data
'''
woe值，iv值
'''
# woe值，iv值计算方法
def bin_woe(tar, var, type=None, bin_type='frequency', bins=None, bins_key=None):
    if bin_type == 'frequency':
        # 等频分箱
        if type == 'c':
            temp_dict = dict(var.value_counts())
            bins_temp = 1
            sum = 0
            all_sum = var.shape[0]
            flag = 0
            bins_key = 2
            replace_key = 'big'
            for k, v in temp_dict.items():
                if 'big' == replace_key and flag == 1:
                    replace_key = k
                if flag == 1:
                    var.replace(k, replace_key, inplace=True)
                sum += v
                if (1 - (sum / all_sum)) <= (1 / (bins_temp + bins_key)):
                    flag = 1
                bins_temp += 1
        if type == 'n' and bins is not None:
            msheet = pd.DataFrame({tar.name: tar, var.name: var, 'var_bins': pd.qcut(var, bins, duplicates='drop')})
            grouped = msheet.groupby(['var_bins'])
        elif type == 'c':
            msheet = pd.DataFrame({tar.name: tar, var.name: var})
            grouped = msheet.groupby([var.name])
    elif bin_type == 'range' and bins is not None:
        # 等距分箱
        msheet = pd.DataFrame({tar.name: tar, var.name: var, 'var_bins': pd.cut(var, bins,right=False, duplicates='drop')})
        grouped = msheet.groupby(['var_bins'])
    elif bin_type == 'key' and bins_key is not None:
        # 指定分箱
        msheet = pd.DataFrame({tar.name: tar, var.name: var, 'var_bins': pd.cut(var, bins_key,right=False, duplicates='drop')})
        grouped = msheet.groupby(['var_bins'])
    # 计算
    total_bad = tar.sum()
    total_good = tar.count() - tar.sum()
    totalRate = total_bad / total_good
    groupBad = grouped.sum()[tar.name]
    groupTotal = grouped.count()[tar.name]
    groupGood = groupTotal - groupBad
    # 这里需要处理一个分箱如果bad或者good为0的情况，可以做一个平滑处理
    for i in range(0, len(groupBad)):
        if groupBad.iloc[i] == 0 or groupGood.iloc[i] == 0:
            groupBad.iloc[i] += 0.5
            groupGood.iloc[i] += 0.5
    groupRate = groupBad / groupGood
    # woe计算公式
    woe = np.log(groupRate / totalRate)
    # iv计算公式
    iv = np.sum((groupBad / total_bad - groupGood / total_good) * woe)
    dictmap = {}
    for x in woe.index:
        dictmap[x] = woe.ix[x]
    new_var, cut = var.map(dictmap), woe.index

    return woe.tolist(), iv, cut, new_var

'''
数据初步查看
'''
def vis_data(data):
    print(data.columns.tolist())
    for item in data.columns.tolist():
        print(item)
        #print("max",data[item].max())
        #print("min",data[item].min())
        print("缺失率",1-(data[item].count()/data.shape[0]))
        temp_lis = data[item].unique().tolist()
'''
cat 编码
'''
def categlory_encoder(fea):
    nf_dict = dict()
    nf_dict['UNK'] = -1
    idx = 0
    for value in fea.values:
        if str(value) not in nf_dict:
            nf_dict[str(value)] = idx
            idx += 1
        else:
            continue
    fea = fea.astype(str).map(nf_dict)
    return fea


'''
稀疏特征类
'''
class SparseFeat(namedtuple('SparseFeat', ['name', 'dimension', 'use_hash', 'dtype', 'embedding_name', 'embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, use_hash=False, dtype="int32", embedding_name=None, embedding=True):
        if embedding and embedding_name is None:
            embedding_name = name
        return super(SparseFeat, cls).__new__(cls, name, dimension, use_hash, dtype, embedding_name, embedding)

'''
稠密特征类
'''
class DenseFeat(namedtuple('SparseFeat', ['name', 'dimension', 'use_hash', 'dtype', 'embedding_name', 'embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, use_hash=False,dtype="float32", embedding_name=None, embedding=True):
        if embedding and embedding_name is None:
            embedding_name = name
        return super(DenseFeat, cls).__new__(cls, name, dimension, use_hash, dtype, embedding_name, embedding)

'''
序列特征类
'''
class VarLenSparseFeat(namedtuple('VarLenFeat',
                                  ['name', 'dimension', 'maxlen', 'combiner', 'use_hash', 'dtype', 'embedding_name',
                                   'embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, maxlen, combiner="mean", use_hash=False, dtype="float32", embedding_name=None,
                embedding=True):
        if embedding_name is None:
            embedding_name = name
        return super(VarLenSparseFeat, cls).__new__(cls, name, dimension, maxlen, combiner, use_hash, dtype,
                                                    embedding_name, embedding)
'''
hash 层
'''
class Hash(keras.layers.Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    # 只有0会被设置成0。其它值被设置成 1到 num_buckets之间的
    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        # 自定义层权重
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        # 操作
        if x.dtype != tf.string:
            x = tf.as_string(x, )
        hash_x = tf.string_to_hash_bucket_fast(x, self.num_buckets if not self.mask_zero else self.num_buckets - 1,
                                               name=None)  # weak hash
        if self.mask_zero:
            mask_1 = tf.cast(tf.not_equal(x, "0"), 'int64')
            mask_2 = tf.cast(tf.not_equal(x, "0.0"), 'int64')
            mask = mask_1 * mask_2
            hash_x = (hash_x + 1) * mask
        return hash_x

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero}
        base_config = super(Hash, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        # concatenate 与 concat作用类似
        return keras.layers.Concatenate(axis=axis)(inputs)
'''
内存优化工具
'''
def reduce_mem_usage(df, prefix='', verbose=True):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('[Memory usage of dataframe is {:.2f} MB]'.format(start_mem))
    for col in tqdm(df.columns, desc='reduce_mem_usage_' + prefix):
        if prefix == '':
            trange(1, desc='processing: ' + col, position=1, bar_format='{desc}')
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('[Memory usage after optimization is: {:.2f} MB]'.format(end_mem))
        print('[Decreased by {:.1f}%]'.format(100 * (start_mem - end_mem) / start_mem))

    return df

'''
auc 计算
'''
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return K.sum(s, axis=0)

'''
mix_auc
'''
def mixed_auc(y_true,y_pred):
    return round(roc_auc_score(y_true[:,0], y_pred[:,0]), 4)



# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP / N


# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP / P


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = {'batch': [], 'epoch': []}
        self.binary_crossentropy = {'batch': [], 'epoch': []}
        self.auc = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_binary_crossentropy = {'batch': [], 'epoch': []}
        self.val_auc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.loss['batch'].append(logs.get('loss'))
        self.binary_crossentropy['batch'].append(logs.get('binary_crossentropy'))
        self.auc['batch'].append(logs.get('auc'))
        self.val_loss['batch'].append(logs.get('loss'))
        self.val_binary_crossentropy['batch'].append(logs.get('binary_crossentropy'))
        self.val_auc['batch'].append(logs.get('auc'))

    def on_epoch_end(self, batch, logs={}):
        self.loss['epoch'].append(logs.get('loss'))
        self.binary_crossentropy['epoch'].append(logs.get('binary_crossentropy'))
        self.auc['epoch'].append(logs.get('auc'))
        self.val_loss['epoch'].append(logs.get('loss'))
        self.val_binary_crossentropy['epoch'].append(logs.get('binary_crossentropy'))
        self.val_auc['epoch'].append(logs.get('auc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.loss[loss_type]))
        plt.figure()
        # binary_crossentropy
        plt.plot(iters, self.binary_crossentropy[loss_type], label='train binary_crossentropy')
        # loss
        plt.plot(iters, self.loss[loss_type], label='train loss')
        # auc
        plt.plot(iters, self.auc[loss_type], label='train auc')
        if loss_type == 'epoch':
            # val_binary_crossentropy
            plt.plot(iters, self.binary_crossentropy[loss_type], label='val binary_crossentropy')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], label='val loss')
            # val_auc
            plt.plot(iters, self.val_auc[loss_type], label='val auc')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('binary_crossentropy-loss-auc')
        plt.legend(loc="upper right")
        plt.show()
'''
序列池化层
'''
class SequencePoolingLayer(Layer):
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.

      Input shape
        - A list of two  tensor [seq_value,seq_len]

        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``

        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **mode**:str.Pooling operation to be used,can be sum,mean or max.

        - **supports_masking**:If True,the input need to support masking.
    """

    def __init__(self, mode='mean', supports_masking=False, **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = 1e-8
        super(SequencePoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = input_shape[0][1].value
        super(SequencePoolingLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError("When supports_masking=True, input must support masking")
            uiseq_embed_list = seq_value_len_list
            mask = tf.to_float(mask)
            user_behavior_length = tf.reduce_sum(mask, axis=-1, keep_dims=True)
            mask = tf.expand_dims(mask, axis=2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list

            mask = tf.sequence_mask(user_behavior_length, self.seq_len_max, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = uiseq_embed_list.shape[-1]

        mask = tf.tile(mask, [1, 1, embedding_size])
        uiseq_embed_list *= mask
        hist = uiseq_embed_list
        if self.mode == "max":
            return tf.reduce_max(hist, 1, keep_dims=True)

        hist = tf.reduce_sum(hist, 1, keep_dims=False)
        if self.mode == "mean":
            hist = tf.div(hist, user_behavior_length + self.eps)

        hist = tf.expand_dims(hist, axis=1)
        return hist

    def compute_output_shape(self, input_shape):
        if self.supports_masking:
            return None, 1, input_shape[-1]
        else:
            return None, 1, input_shape[0][-1]

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(SequencePoolingLayer, self).get_config()
        # noinspection PyTypeChecker
        return dict(list(base_config.items()) + list(config.items()))

'''
延迟转化loss 
'''
'''
def my_custom_loss(y_true,y_pred):
    t_loss = (-1)*(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
    return K.mean(t_loss)
'''
def mixed_loss(input,pred):

    y_true = input[:,0]
    delay = input[:,1]
    elapse = input[:,2]
    loss2 = K.mean((-1)*(y_true * tf.math.log(pred[:,0]) + (1 - y_true) * K.log(1 - pred[:,0])))
    loss = K.mean(-1 * y_true * (tf.math.log(pred[:,0]) + tf.math.log(pred[:,0]) - pred[:,0] * delay) \
                      -(1-y_true) * (tf.math.log(1 - pred[:,0] + pred[:,0] * tf.exp(-1 * pred[:,0] * elapse))), axis=-1)
    
    return 0.1*loss+0.9*loss2