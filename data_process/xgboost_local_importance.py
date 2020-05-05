import json
import pandas as pd
import numpy as np

def fcc_f(model_json,x,data,fc_dict):
    if 'leaf' in model_json.keys():
        return fc_dict
    split_fea = model_json['split']
    split_condition = model_json['split_condition']
    node_yes = model_json['yes']
    node_no = model_json['no']
    node_missing = model_json['missing']
    temp_json = model_json['children']
    #原来的比例
    pp = (data[data['label']==x['label']].shape[0])/data.shape[0]
    if x[split_fea] < split_condition:
        key = node_yes
        data = data.loc[data[split_fea]<split_condition,:]
    elif x[split_fea] >= split_condition:
        key = node_no
        data = data.loc[data[split_fea] < split_condition, :]
    else:
        key = node_missing
        data = data.loc[pd.isna(data[split_fea]), :]
    pt = (data[data['label']==x['label']].shape[0])/data.shape[0]

    if split_fea in fc_dict.keys():
        fc_dict[split_fea] += (pt-pp)
    else:
        fc_dict[split_fea] = (pt-pp)
    for item in temp_json:
        node_id = item['nodeid']
        if key == node_id:
            return fcc_f(item,x,data,fc_dict)




def fcf_f(model_json,fc_dict):
    if 'leaf' in model_json.keys():
        return
    split_fea = model_json['split']
    split_condition = model_json['split_condition']

    if split_fea in fc_dict.keys():
        fc_dict[split_fea].append(split_condition)
    else:
        fc_dict[split_fea] = []
        fc_dict[split_fea].append(split_condition)
    model_json = model_json['children']
    for item in model_json:
        fcf_f(item,fc_dict)




def fcf_f1(model_json,x,data):
    fcff_temp = {}
    for item in model_json:
        fcf_f(item,fcff_temp)
    fcff_temp1 = {}
    for k,v in fcff_temp.items():
        v.sort()
        a = set(v)
        fcff_temp1[k] = a
    fccf_dict = {}
    pp = (data[data['label']==x['label']].shape[0])/data.shape[0]
    for k,v in fcff_temp1.items():
        v = list(v)
        for i in range(0,len(v)):
            if x[k]<v[i]:
                if i==0:
                   pt = data[(data[k]<v[i])&(data['label']==x['label'])].shape[0]/data[data[k]<v[i]].shape[0]
                else:
                   pt = data[(data[k] < v[i])&(data[k]>=v[i-1])&(data['label'] == x['label'])].shape[0] /\
                        data[(data[k] < v[i])&(data[k]>=v[i-1])].shape[0]
                break;
        if i == len(v):
            pt = data[(data[k] >=v[i-1])  & (data['label'] == x['label'])].shape[0] / \
                 data[data[k] < v[i]].shape[0]
        temp = {}
        temp['value'] = pt-pp
        if i == 0:
           temp['left'] = np.inf
        else:
           temp['left'] = v[i-1]
        if i==len(v):
            temp['right'] = np.inf
        else:
            temp['right'] = v[i]
        fccf_dict[k] = temp
    return fccf_dict

'''
先求区间
'''
def foc_f(model_json,x,fc_dict):
    if 'leaf' in model_json.keys():
        return fc_dict
    split_fea = model_json['split']
    split_condition = model_json['split_condition']
    node_yes = model_json['yes']
    node_no = model_json['no']
    node_missing = model_json['missing']
    temp_json = model_json['children']
    if x[split_fea] < split_condition:
        key = node_yes
        right = split_condition
    elif x[split_fea] >= split_condition:
        key = node_no
        left = split_condition
    else:
        key = node_missing
    if split_fea in fc_dict.keys():
        if key == node_no:
            if 'left' in fc_dict[split_fea].keys():
                if fc_dict[split_fea]['left']<left:
                   fc_dict[split_fea]['left'] = left
            else:
                fc_dict[split_fea]['left'] = left
        elif key==node_yes:
            if 'right' in fc_dict[split_fea].keys():
                if fc_dict[split_fea]['right']>right:
                   fc_dict[split_fea]['right'] = right
            else:
                fc_dict[split_fea]['right'] = right
    else:
        temp = {}
        if key == node_no:
            temp['left'] = left
        elif key == node_yes:
            temp['right'] = right
        fc_dict[split_fea] = temp
    for item in temp_json:
        node_id = item['nodeid']
        if key == node_id:
            return foc_f(item, x, fc_dict)
def foc_f1(x,fc_dict,data):
    for k,v in fc_dict.items():
        if 'left' in v.keys() and 'right' in v.keys():
            #x.value一定会在区间里面否则路径无法走通
            if data[(data[k]>=x[k])&(data[k]<v['right'])].shape[0]==0:
                p = data[(data[k]<x[k])&(data[k]>=v['left'])&(data['label']==x['label'])].shape[0]/\
                    data[(data[k]<x[k])&(data[k]>=v['left'])].shape[0]
            elif data[(data[k]<x[k])&(data[k]>=v['left'])].shape[0]==0:
                p = data[(data[k]<x[k])&(data[k]>=v['left'])&(data['label']==x['label'])].shape[0]/\
                    data[(data[k]<x[k])&(data[k]>=v['left'])].shape[0]
            else:
                p = max(data[(data[k]>=x[k])&(data[k]<v['right'])&(data['label']==x['label'])].shape[0]/
                    data[(data[k]>=x[k])&(data[k]<v['right'])].shape[0],data[(data[k]<x[k])&(data[k]>=v['left'])&(data['label']==x['label'])].shape[0]/
                    data[(data[k]<x[k])&(data[k]>=v['left'])].shape[0])
        elif 'left' in v.keys():
            if data[(data[k] >= x[k])].shape[0]==0:
                p =  data[(data[k] < x[k]) & (data[k] >= v['left']) & (data['label'] == x['label'])].shape[0] /\
                     data[(data[k] < x[k]) & (data[k] >= v['left'])].shape[0]
            elif data[(data[k] < x[k]) & (data[k] >= v['left'])].shape[0]==0:
                p = data[(data[k] >= x[k])  & (data['label'] == x['label'])].shape[0] /\
                    data[(data[k] >= x[k])].shape[0]
            else:
                p = max(data[(data[k] >= x[k])  & (data['label'] == x['label'])].shape[0] /
                    data[(data[k] >= x[k])].shape[0],
                    data[(data[k] < x[k]) & (data[k] >= v['left']) & (data['label'] == x['label'])].shape[0] /
                    data[(data[k] < x[k]) & (data[k] >= v['left'])].shape[0])
        elif 'right' in v.keys():
            if data[(data[k] < x[k])].shape[0]==0:
                p = data[(data[k] >= x[k]) & (data[k] < v['right']) & (data['label'] == x['label'])].shape[0] /\
                    data[(data[k] >= x[k]) & (data[k] < v['right'])].shape[0]
            elif data[(data[k] >= x[k]) & (data[k] < v['right'])].shape[0]==0:
                p = data[(data[k] < x[k]) & (data['label'] == x['label'])].shape[0] /\
                    data[(data[k] < x[k])].shape[0]
            else:
                p = max(data[(data[k] < x[k]) & (data['label'] == x['label'])].shape[0] /
                    data[(data[k] < x[k])].shape[0],
                    data[(data[k] >= x[k]) & (data[k] < v['right']) & (data['label'] == x['label'])].shape[0] /
                    data[(data[k] >= x[k]) & (data[k] < v['right'])].shape[0])
        fc_dict[k] = p
    return fc_dict

def fof_f(model_json,x,data):
    fcff_temp = {}
    for item in model_json:
        fcf_f(item, fcff_temp)
    ###求出x所在区间。求max即可
    print(fcff_temp)
    fcff_temp1 = {}
    for k,v in fcff_temp.items():
        v.sort()
        a = set(v)
        fcff_temp1[k] = a
    foff_dict = {}
    for k,v in fcff_temp1.items():
        v = list(v)
        for i in range(0,len(v)):
            if x[k]<v[i]:
                if i==0:
                   if data[data[k]<x[k]].shape[0]==0:
                       p = data[(data[k]>=x[k])&(data[k]<v[i])& (data['label'] == x['label'])].shape[0] / data[(data[k]>=x[k])
                           &(data[k]<v[i])].shape[0]
                   elif data[(data[k]>=x[k])&(data[k]<v[i])].shape[0]==0:
                       p = data[(data[k]<x[k])&(data['label']==x['label'])].shape[0]/data[data[k]<x[k]].shape[0]
                   else:
                       p = max(data[(data[k]<x[k])&(data['label']==x['label'])].shape[0]/data[data[k]<x[k]].shape[0],
                           data[(data[k]>=x[k])&(data[k]<v[i])&(data['label']==x['label'])].shape[0]/data[(data[k]>=x[k])
                           &(data[k]<v[i])].shape[0])
                else:
                   if data[(data[k] < x[k])&(data[k]>=v[i-1])].shape[0]==0:
                       p = data[(data[k] < v[i]) & (data[k] >= x[k]) & (data['label'] == x['label'])].shape[0]\
                           /data[(data[k] < v[i]) & (data[k] >= x[k])].shape[0]
                   elif data[(data[k] < v[i]) & (data[k] >= x[k])].shape[0]:
                       p = data[(data[k] < x[k])&(data[k]>=v[i-1])&(data['label'] == x['label'])].shape[0] /\
                        data[(data[k] < x[k])&(data[k]>=v[i-1])].shape[0]
                   else:
                       p = max(data[(data[k] < x[k])&(data[k]>=v[i-1])&(data['label'] == x['label'])].shape[0] /\
                        data[(data[k] < x[k])&(data[k]>=v[i-1])].shape[0],data[(data[k] < v[i]) & (data[k] >= x[k]) & (data['label'] == x['label'])].shape[0]
                           /data[(data[k] < v[i]) & (data[k] >= x[k])].shape[0])
                break;
        if i == len(v):
            if data[(data[k] < v[i])&(data[k]<x[k])].shape[0]==0:
                p = data[(data[k]>=x[k])&(data['label']==x['label'])].shape[0]/data[data[k]>=x[k]].shape[0]
            elif data[data[k]>=x[k]].shape[0]==0:
                p = data[(data[k] >=v[i-1]) &(data[k]<x[k]) & (data['label'] == x['label'])].shape[0] /\
                    data[(data[k] < v[i])&(data[k]<x[k])].shape[0]
            else:
                p = max(data[(data[k] >=v[i-1]) &(data[k]<x[k]) & (data['label'] == x['label'])].shape[0] / \
                    data[(data[k] < v[i])&(data[k]<x[k])].shape[0],data[(data[k]>=x[k])&(data['label']==x['label'])].shape[0]/data[data[k]>=x[k]].shape[0])
        foff_dict[k] = p
    return foff_dict


model_txt = json.load(open('../xgb_model_file/xgb_model_txt','r'))

print(model_txt)


data = pd.read_csv('../xgb_model_file/test_xgb_local.csv')


x = data.loc[0,:]
print("特征贡献重要性")
fccf_dict = {}
i = 1
for item in model_txt:
    print("tree"+str(i))
    fccf_dict = fcc_f(item,x,data,fccf_dict)
    print(fccf_dict)
    i+=1
print("特征维度贡献重要性")
fccf_dict = fcf_f1(model_txt,x,data)
print(fccf_dict)

print("特征异常树中贡献重要性")
i = 1
for item in model_txt:
    print('tree'+str(i))
    focf_dict = foc_f(item,x,{})
    print(foc_f1(x, focf_dict, data))
    i+=1
print("特征异常区间贡献重要性")
print(fof_f(model_txt,x,data))










