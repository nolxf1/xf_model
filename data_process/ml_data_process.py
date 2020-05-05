



import pandas as pd

from xf_model.data_process.common_data_process import common_data_process

class ml_data_process(common_data_process):

    def __init__(self, config):
        super(ml_data_process,self).__init__(config)
        self.train_config = config['train_config']
    def get_xgboost_train_input(self):
        
        self.deal_nan()
        self.categlory_encoder()
        self.one_hot()
        par = self.train_config['par']
        train_data = self.data.loc[self.data['leads_dt']<par,:]
        test_data = self.data.loc[self.data['leads_dt']>=par,:]

        train_data.drop(['leads_dt','driver_id','label'],axis=1,inplace=True)
        test_data.drop(['leads_dt','driver_id','label'],axis=1,inplace=True)

        train_label = self.data.loc[self.data['leads_dt']<par,'label']
        test_label = self.data.loc[self.data['leads_dt']>=par,'label']
        
        print("特征数：",train_data.shape[1])
        print("所使用特征:")
        print(train_data.columns.tolist())
        print("训练集：",train_data.shape[0])
        print("验证集：",test_data.shape[0])
        
        return train_data,test_data,train_label,test_label





