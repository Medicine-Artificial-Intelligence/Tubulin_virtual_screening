import pickle
import os
import numpy as np
import pandas as pd
from Prediction import predict
def biopred(data, ID_col, activity_col, target_dir, target_type = ['A549', 'MDAMB231', 'Tubulin']):
    df = data[[ID_col, activity_col]]
    df.index = df[ID_col].values
    for i in target_type:
        util_dir = target_dir +i +'/utility/'
        while True:
            try:
                pred = predict(materials_path=util_dir,
                                 data = data, activity_col = activity_col,
                                 ID = ID_col)
                break
            except:
                pred = predict(materials_path=util_dir,
                                 data = data, activity_col = 'pChEMBL',
                                 ID = ID_col)
                break
        data_pred = pred.predict()
        data_pred.index = data_pred['Index'].values
        data_pred.drop(['Index'], axis =1, inplace = True)
        data_pred.rename(columns ={'Probability':f'Proba_{i}', 'Predict':f'Pred_{i}'}, inplace = True)
        df = pd.concat([df,data_pred], axis =1)
    return df