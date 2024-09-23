import sys

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

import joblib


def join_datasets(dataset1, dataset2, path):
    df_automl = pd.read_csv(path+dataset1, sep=';')
    df_random = pd.read_csv(path+dataset2, sep=';')

    df = pd.concat([df_automl, df_random], ignore_index=True)
    
    return df


def get_dataset(df, target1, target2):
    print(f'Dados originais: {df.shape}')

    flag = (df[target1].isna())|(df[target2].isna())
    df = df[~flag]
    print(f'Dados sem targets NaN: {df.shape}')
    
    df.iloc[:, :-2] = df.iloc[:, :-2] + 1

    df = df.fillna(0)

    X = df.iloc[:, :-2]
    y = df[[target1, target2]]

    return X, y


def get_metrics(y_test, y_pred): # series
    r2 = metrics.r2_score(y_test, y_pred)
    print(f'R quadrado (R2): {r2}')

    MSE = metrics.mean_squared_error(y_test,y_pred)
    print(f'Erro quadrático médio (MSE): {MSE}')

    RMSE = metrics.mean_squared_error(y_test,y_pred,squared=False)
    print(f'Raiz do erro quadrático médio (RMSE): {RMSE}')

    MAE = metrics.mean_absolute_error(y_test,y_pred)
    print(f'Erro absoluto médio (MAE): {MAE}\n')

    return r2, MSE, RMSE, MAE


# model 1 = Macro F-score
# model 2 = Log Model Size


if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        project = sys.argv[1]   
        meta_dataset = sys.argv[2]   
    else:
        project = ''
        meta_dataset = 'flags'
        
    meta_dataset1 = meta_dataset+'-automl.csv'
    meta_dataset2 = meta_dataset+'-random.csv'
    path_models = project + 'models/'
    path_meta_datasets = project + 'meta_datasets/'
    
    # Obtém X e y
    df = join_datasets(meta_dataset1, meta_dataset2, path_meta_datasets)
    X, y = get_dataset(df.copy(), 'F1 (macro averaged by label)', 'Model Size')
    y['Log Model Size'] = np.log(y['Model Size']) # log
    
    # treina regressor - macro f-score
    regressor1 = RandomForestRegressor()
    regressor1.fit(X, y['F1 (macro averaged by label)'])
    joblib.dump(regressor1, path_models+'/model-'+meta_dataset+'-obj1.sav') 

    # treina regressor - log model size
    regressor2 = RandomForestRegressor()
    regressor2.fit(X, y['Log Model Size'])
    joblib.dump(regressor2, path_models+'/model-'+meta_dataset+'-obj2.sav')