
import pandas as pd
from xf_model.common import uitls
import warnings
warnings.filterwarnings("ignore")


class common_data_process:

    def __init__(self, config):
        self.fea_config = config['fea_config']
        self.data = pd.read_csv(config['data_path'])
    #空值
    def deal_nan(self):
        print('*'*10,'deal_nan','*'*10)
        nan_config = self.fea_config['nan_config']
        switch = {'mean': uitls.fill_nan_by_mean,
                  'mode': uitls.fill_nan_by_mode,
                  'mid': uitls.fill_nan_by_mid,
                  }
        for key,nan_c in nan_config.items():
            if nan_c['nan_c'] in switch.keys():
                print('=' * 10, nan_c['nan_c']+'>'+key)
                self.data[key] = switch.get(nan_c['nan_c'])(self.data[key])
            else:
                print('=' * 10, nan_c['nan_c']+':'+str(nan_c['nan_key'])+'>'+key)
                self.data[key] = uitls.fill_nan_by_key(self.data[key],nan_c['nan_key'])
    #异常值
    def deal_yc(self):
        '''
        需后续优化
        '''
        yc_config = self.fea_config['yc_config']
        switch = {'std_mean':uitls.check_specail_by_std_mean,
                  'quantile':uitls.check_specail_by_quantile,
                 }
        for key,yc_c in yc_config.items():
            self.data[key] = switch.get(yc_c['yc_c'])(self.data[key])
        #self.deal_nan()
    #分桶配置
    def deal_bin(self):
        bin_config = self.fea_config['bin_config']
        for key,bins in bin_config.items():
            self.data[key] = uitls.bin_featrue(self.data[key],bins)
    # one_hot
    def one_hot(self):
        print('*' * 10, 'one_hot', '*' * 10)
        one_hot_config = self.fea_config['one_hot_config']
        for key in one_hot_config:
            fea_data = uitls.drop_axis(self.data[key],0.8)
            print('=' * 10, 'input dim'+str(len(fea_data.unique().tolist()))+'>' + key)
            fea_data = uitls.oneHot(fea_data,fea_data.unique().tolist(),key)
            self.data = self.data.join(fea_data)
            self.data.drop(key, axis=1, inplace=True)
    # log
    def deal_log(self):
        log_config = self.fea_config['log_config']
        for key in log_config:
            self.data[key] = uitls.log_featrue(self.data[key])
    # min_max
    def deal_min_max(self):
        print('*' * 10, 'min_max', '*' * 10)
        min_max_config = self.fea_config['min_max_config']
        for key in min_max_config:
            print('=' * 10, '>' + key)
            self.data[key] = uitls.min_max_feature(self.data[key])
    # standard
    def deal_standard(self):
        standard_config = self.fea_config['standard_config']
        for key in standard_config:
            self.data[key] = uitls.standard_scaler_feature(self.data[key])

    def categlory_encoder(self):
        print('*' * 10, 'categlory', '*' * 10)
        sparse_features_config = self.fea_config['sparse_fea_config']
        for key in sparse_features_config:
            print('=' * 10, '>' + key)
            self.data[key] = uitls.categlory_encoder(self.data[key])