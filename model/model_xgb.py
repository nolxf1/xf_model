import pickle
import pandas as pd
import xgboost as xgb
from sklearn import metrics
import numpy as np

class BASE_XGB:

    def __init__(self,config):
        self.roundNum = config['round_num']
        self.params = config['xgb_params']
        self.model = None
        self.model_file = config['model_file']
    def train(self,data,label,params=None):
        dtrain = xgb.DMatrix(data.astype('float').values, label, feature_names=data.columns.values.tolist())
        self.model = xgb.train(self.params, dtrain, self.roundNum)
        self.saveModel(self.model_file)
        self.saveTxtModel(self.model_file + '_txt')
    def predict(self, data, dropList=None, chunkSize=0):
        self.loadModel(self.model_file)
        if chunkSize == 0:
            results = self.model.predict(
                xgb.DMatrix(data.astype('float').values, feature_names=data.columns.values.tolist()))
        else:
            results = []
            for chunk in data:
                chunk.drop(dropList, inplace=True, axis=1)
                result = self.model.predict(
                    xgb.DMatrix(chunk.astype('float').values, feature_names=chunk.columns.values.tolist()))
                results = results + [round(i, 4) for i in result]
        return results
    def saveModel(self, modelFile):
        f = open(modelFile, 'wb')
        pickle.dump(self.model, f, protocol=2)
        f.close()

    def saveTxtModel(self, modelFile):
        self.model.dump_model(modelFile,dump_format="json")

    def loadModel(self, modelFile):
        f = open(modelFile, 'rb')
        self.model = pickle.load(f)
        f.close()

    def getFeatureImportance(self, modelFile, importanceType='weight'):
        self.loadModel(modelFile)
        xgb_fea_imp = pd.DataFrame(list(self.model.get_score(importance_type=importanceType).items()),
                                   columns=['feature', 'importance']).sort_values('importance', ascending=False)
        return xgb_fea_imp
class XGB_MODEL(BASE_XGB):

    def __init__(self, config):
        super(XGB_MODEL,self).__init__(config)

    def test(self, data, label):
        self.loadModel(self.model_file)
        results = self.model.predict(
            xgb.DMatrix(data.astype('float').values, feature_names=data.columns.values.tolist()))

        self.modelMetrics(label, results)
        return results

    def modelMetrics(self, label, results):
        results = [round(i, 4) for i in results]
        auc_score = metrics.roc_auc_score(label, results)
        print('MODEL METRICS---AUC: %.4f' % auc_score)
        for i in range(0, 100, 1):
            self.precision_recall(label, results, float(i / 100.0))

    def precision_recall(self, label, results, threshold):
        binary_res = [1 if i >= threshold else 0 for i in results]
        binary_label =  np.array(label)
        if len(binary_res) != len(binary_label):
            raise RuntimeError("unequal size!")
        else:
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i in range(len(binary_res)):
                if binary_res[i] == 1 and binary_label[i] == 1:  # true and predict ture
                    tp = tp + 1
                elif binary_res[i] == 1 and binary_label[i] == 0:  # false but predict true
                    fp = fp + 1
                elif binary_res[i] == 0 and binary_label[i] == 1:  # true but predict false
                    fn = fn + 1
                elif binary_res[i] == 0 and binary_label[i] == 0:  # false and predict false
                    tn = tn + 1
            if (tp + fp) != 0 and (tp + fn) != 0:
                precision = tp / float(tp + fp)
                recall = tp / float(tp + fn)
                print('MODELMETRICS---THRESHOLD: %.2f PRECISION: %.4f RECALL: %.4f' % (threshold, precision, recall))

class XGB_MODEL_REG(BASE_XGB):

    def __init__(self, config):
        super(XGB_MODEL, self).__init__(config)

    def test(self, modelFile, data, label):
        self.loadModel(modelFile)
        results = self.model.predict(xgb.DMatrix(data.astype('float').values, feature_names = data.columns.values.tolist()))

        self.modelMetrics(label, results)
        return results

    def modelMetrics(self, label, results):
        results = [round(i,4) for i in results]
        mae = metrics.mean_absolute_error(label, results)
        print ('MODEL METRICS---MAE: %.4f' % mae)



