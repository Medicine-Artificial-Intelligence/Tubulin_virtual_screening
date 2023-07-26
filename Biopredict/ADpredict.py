import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
class Applicability_domain_LOF:
    def __init__(self, data, ad_dir, ID_col ='ID', activity_col='pChEMBL Value', 
                 target_type_list = ['MDAMB231', 'Tubulin','A549']):
        self.data = data
        self.ID_col = ID_col
        self.activity_col = activity_col
        self.ad_dir = ad_dir
        self.target_type_list = target_type_list
        self.X = self.data.drop([self.activity_col, self.ID_col],axis =1)
    
    def LOF(self, base_data, target_type):
        clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.01)
        clf.fit(base_data)
        y_pred_outliers = pd.DataFrame(clf.predict(self.X), columns = [f'AD_{target_type}'])
        y_pred_outliers.loc[y_pred_outliers.values==1]='In'
        y_pred_outliers.loc[y_pred_outliers.values==-1]='Out'
        self.data = pd.concat([self.data,y_pred_outliers], axis =1)
        #return data
    
    def fit(self):
        for i in self.target_type_list:
            #print(i)
            base_data = pd.read_csv(self.ad_dir + f'/ad_{i}.csv', index_col=0).drop([self.activity_col], axis =1)
            #display(base_data.shape)
            self.LOF(base_data=base_data, target_type=i)
        display(self.data.head(5))
            