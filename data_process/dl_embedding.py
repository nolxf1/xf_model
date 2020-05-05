from collections import OrderedDict
from keras.initializers import RandomNormal, TruncatedNormal
from keras.layers import Concatenate, Dense, Embedding, Input, add, Flatten,multiply,RepeatVector,Reshape
from keras.regularizers import l2
from xf_model.common.uitls import SequencePoolingLayer
from xf_model.common.uitls import Hash, concat_fun, SparseFeat, DenseFeat, VarLenSparseFeat

def get_fixlen_feature_names(feature_columns,prefix=''):
    features = build_input_features(feature_columns, include_varlen=False, include_fixlen=True,prefix='')
    return features.keys()

def get_varlen_feature_names(feature_columns,prefix=''):
    features = build_input_features(feature_columns, include_varlen=True, include_fixlen=False,prefix='')
    return features.keys()

def build_input_features(feature_columns, include_varlen=True, mask_zero=True, prefix='', include_fixlen=True):
    input_features = OrderedDict()
    if include_fixlen:
        for fc in feature_columns:
            if isinstance(fc, SparseFeat):
                input_features[fc.name] = Input(shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
            elif isinstance(fc, DenseFeat):
                input_features[fc.name] = Input(shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
    if include_varlen:
        for fc in feature_columns:
            if isinstance(fc, VarLenSparseFeat):
                input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + 'seq_' + fc.name, dtype=fc.dtype)
        if not mask_zero:
            for fc in feature_columns:
                input_features[fc.name+"_seq_length"] = Input(shape=(1,), name=prefix + 'seq_length_' + fc.name)
                input_features[fc.name+"_seq_max_length"] = fc.maxlen

    return input_features


def create_sparse_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, embedding_size, init_std, seed, l2_reg, prefix='sparse_', seq_mask_zero=True):

    feat_embed_dict = {}
    if sparse_feature_columns and len(sparse_feature_columns):
        for feat in sparse_feature_columns:
            feat_embed_dict[feat.embedding_name] = Embedding(feat.dimension, embedding_size,
                                                       embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std, seed=seed),
                                                       embeddings_regularizer=l2(l2_reg),
                                                       name=prefix + '_emb_'  + feat.name)


    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            feat_embed_dict[feat.embedding_name] = Embedding(feat.dimension, embedding_size,
                                                              embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std, seed=seed),
                                                              embeddings_regularizer=l2(l2_reg),
                                                              name=prefix + '_seq_emb_' + feat.name,
                                                              mask_zero=seq_mask_zero)

    return feat_embed_dict



def create_dense_embedding_dict(dense_feature_columns, embedding_size, init_std, seed, l2_reg, prefix='sparse_', seq_mask_zero=True):

    feat_embed_dict = {}

    if dense_feature_columns and len(dense_feature_columns) > 0:
        for feat in dense_feature_columns:
            feat_embed_dict[feat.embedding_name] = Dense(embedding_size, activation=None, use_bias=False,kernel_initializer=RandomNormal(mean=0.0, stddev=init_std, seed=seed),
                                                                      kernel_regularizer=l2(l2_reg),name=prefix+'_emb_'+feat.name)
    return feat_embed_dict
def sparse_embedding_lookup(sparse_embedding_dict,sparse_input_dict,sparse_feature_columns,return_feat_list=(), mask_feat_list=()):
    embedding_vec_list = []
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if len(return_feat_list) == 0  or feature_name in return_feat_list and fc.embedding:
            if fc.use_hash:
                lookup_idx = Hash(fc.dimension,mask_zero=(feature_name in mask_feat_list))(sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]
            embedding_vec_list.append(sparse_embedding_dict[embedding_name](lookup_idx))
    return embedding_vec_list


def dense_embedding_lookup(dense_embedding_dict,dense_input_dict,dense_feature_columns,embedding_size,return_feat_list=(), mask_feat_list=()):
    embedding_vec_list = []
    for fc in dense_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if len(return_feat_list) == 0  or feature_name in return_feat_list and fc.embedding:
            lookup_idx = dense_input_dict[feature_name]
            embedding_vec_list.append(Reshape((-1,embedding_size))(dense_embedding_dict[embedding_name](lookup_idx)))
    return embedding_vec_list

def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.dimension, mask_zero=True)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)

    return varlen_embedding_vec_dict

def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns):
    pooling_vec_list = []
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = feature_name + '_seq_length'
        if feature_length_name in features:
            vec = SequencePoolingLayer(combiner, supports_masking=False)([embedding_dict[feature_name], features[feature_length_name]])
        else:
            vec = SequencePoolingLayer(combiner, supports_masking=True)(embedding_dict[feature_name])
        pooling_vec_list.append(vec)
    return pooling_vec_list

def get_dense_input(features,feature_columns):
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        dense_input_list.append(features[fc.name])
    return dense_input_list

def embedding_input_from_feature_columns(features, feature_columns, embedding_size, l2_reg, init_std, seed,prefix='', seq_mask_zero=True):
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []

    sparse_embedding_dict = create_sparse_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, embedding_size,
                                            init_std, seed, l2_reg, prefix=prefix + 'sparse',
                                            seq_mask_zero=seq_mask_zero)
    sparse_embedding_list = sparse_embedding_lookup(sparse_embedding_dict, features, sparse_feature_columns)

    sequence_embed_dict = varlen_embedding_lookup(sparse_embedding_dict, features, varlen_sparse_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, varlen_sparse_feature_columns)
    sparse_embedding_list += sequence_embed_list

    dense_embedding_dict = create_dense_embedding_dict(dense_feature_columns, embedding_size,
                                            init_std, seed, l2_reg, prefix=prefix + 'dense',
                                            seq_mask_zero=seq_mask_zero)
    dense_embedding_list = dense_embedding_lookup(dense_embedding_dict,features,dense_feature_columns,embedding_size)


    return sparse_embedding_list, dense_embedding_list



def get_linear_logit(features, feature_columns, units=1, l2_reg=0, init_std=0.0001, seed=1024, prefix='linear'):
    linear_emb_list = [embedding_input_from_feature_columns(features, feature_columns, 1, l2_reg, init_std, seed, prefix=prefix + str(i))[0] for
        i in range(units)]

    dense_input_list = [embedding_input_from_feature_columns(features, feature_columns, 1, l2_reg, init_std, seed, prefix=prefix + str(i))[1] for
        i in range(units)]

    if len(linear_emb_list[0]) > 1:
        linear_term = concat_fun([add(linear_emb) for linear_emb in linear_emb_list])
    elif len(linear_emb_list[0]) == 1:
        linear_term = concat_fun([linear_emb[0] for linear_emb in linear_emb_list])
    else:
        linear_term = None
    linear_logit = linear_term

    if len(dense_input_list[0]) > 1:
        dense_term = concat_fun([add(dense_emb) for dense_emb in dense_input_list])
    elif len(dense_input_list[0]) == 1:
        dense_term = concat_fun(([dense_emb[0] for dense_emb in dense_input_list]))
    else:
        dense_term = None
    if linear_logit is None:
        linear_logit = dense_term
    elif dense_term is not None:
        linear_logit = add([linear_term,dense_term])


    return linear_logit


def combined_dnn_input_dense_embed(sparse_embedding_list, dense_embedding_list):
    if len(sparse_embedding_list) > 0 and len(dense_embedding_list) > 0:
        sparse_dnn_input = Flatten()(concat_fun(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_fun(dense_embedding_list))
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_fun(sparse_embedding_list))
    elif len(dense_embedding_list) > 0:
        return Flatten()(concat_fun(dense_embedding_list))
    else:
        raise NotImplementedError





