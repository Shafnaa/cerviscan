import pandas as pd
import xgboost as xgb
import pickle

def __main__(featues: pd.DataFrame):
    model: xgb.XGBClassifier = pickle.load(open('./model/xgb_best', 'rb'))
    
    print("masuk")
    result = model.predict(featues)
    
    return result[0]