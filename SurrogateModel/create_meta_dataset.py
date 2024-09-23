import sys

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

from configuration.Configuration import Configuration

from utils.MetaDatasetUtils import *

from utils.IndividualUtils import IndividualUtils


# get_metrics: creates dataset with median value of k-folds execution
def get_metrics(path_metrics, name_dataset):
    paths = [os.path.join(path_metrics+name_dataset, arq) for arq in os.listdir(path_metrics+name_dataset)]
    files = [file for file in paths if os.path.isfile(file)]
    csvs = [file for file in files if file.lower().endswith(".csv")]
    
    is_first = True
    for csv in csvs:
        df = pd.read_csv(csv, sep=';')
                         
        if is_first == True:
            is_first = False
            df_concat = df
        else:
            df_concat = pd.concat([df_concat, df], ignore_index=True)   
    
    list_unique = []
    for norm in df_concat['normalize'].unique():
        df_norm = df_concat[df_concat['normalize']==norm]
        for m in df_norm['meka'].unique():
            df_aux = df_norm[df_norm['meka'] == m]
            algs_weka = df_aux['weka'].unique()
            
            # dataframe concatenation is storing column names there is some error in the files
            if m == 'meka':
                continue
            
            # AA
            if not isinstance(algs_weka[0], str) and np.isnan(algs_weka[0]):
                serie_fscore = df_aux['F1 (macro averaged by label)']
                list_fscore = list(serie_fscore)
                serie_fscore = serie_fscore[~pd.isna(serie_fscore)]
                list_fscore_aux = np.array(serie_fscore.values, dtype=float)
                
                if len(list_fscore_aux)==0:
                    median1 = std1 = pd.NA
                else:
                    median1 = np.median(list_fscore_aux)
                    std1 = np.std(list_fscore_aux)
              
                if len(list_fscore_aux)<3:
                    l = list(df_aux['F1 (macro averaged by label)'])
                    # print(f'{l} {list_fscore_aux} {std1} {median1}')
                
                serie_size = df_aux['Model Size']
                list_size = list(serie_size)
                serie_size = serie_size[~pd.isna(serie_size)]
                list_size_aux = np.array(serie_size.values, dtype=float)
                
                if len(list_size_aux)==0:
                    median2 = std2 = pd.NA
                else:
                    median2 = np.median(list_size_aux)
                    std2 = np.std(list_size_aux)
    
                l = np.array([norm, m, None, list_fscore, std1, median1, list_size, std2, median2], dtype=object)
                if len(list_unique) == 0:
                    list_unique = l
                else:
                    list_unique = np.vstack((list_unique, l))
                   
            # PT
            else:
              for w in algs_weka:
                df_aux2 = df_aux[df_aux['weka'] == w]
                serie_fscore = df_aux2['F1 (macro averaged by label)']
                list_fscore = list(serie_fscore)
                serie_fscore = serie_fscore[~pd.isna(serie_fscore)]
                list_fscore_aux = np.array(serie_fscore.values, dtype=float)
                
                if len(list_fscore_aux)==0:
                    median1 = std1 = pd.NA
                else:
                    median1 = np.median(list_fscore_aux)
                    std1 = np.std(list_fscore_aux)
                
                if len(list_fscore_aux)<3:
                    l = list(df_aux2['F1 (macro averaged by label)'])
                    # print(f'{l} {list_fscore_aux} {std1} {median1}')
                
                serie_size = df_aux2['Model Size']
                list_size = list(serie_size)
                serie_size = serie_size[~pd.isna(serie_size)]
                list_size_aux = np.array(serie_size.values, dtype=float)
                
                if len(list_size_aux)==0:
                    median2 = std2 = pd.NA
                else:
                    median2 = np.median(list_size_aux)
                    std2 = np.std(list_size_aux)                

                l = np.array([norm, m, w, list_fscore, std1, median1, list_size, std2, median2], dtype=object)
                if len(list_unique) == 0:
                    list_unique = l
                else:
                    list_unique = np.vstack((list_unique, l))   
                
    df_unique = pd.DataFrame(list_unique, columns=['normalize', 'meka', 'weka', 'list_fscore', 'std_fscore', 'median_fscore', 'list_size', 'std_size', 'median_size'])
    
    # bayesnet treats
    set_bayes_net(df_unique)

    return df_unique


def set_bayes_net(dataframe):
    dataframe['weka'] = dataframe.apply(lambda L: str(L.weka).replace('-Q weka.classifiers.bayes.net.search.local.K2 -- -P 1', 
                                                            '-Q \'weka.classifiers.bayes.net.search.local.K2 -- -P 1\''), axis=1)
    
    dataframe['weka'] = dataframe.apply(lambda L: str(L.weka).replace('-Q weka.classifiers.bayes.net.search.local.HillClimber -- -P 1', 
                                                            '-Q \'weka.classifiers.bayes.net.search.local.HillClimber -- -P 1\''), axis=1)
    
    dataframe['weka'] = dataframe.apply(lambda L: str(L.weka).replace('-Q weka.classifiers.bayes.net.search.local.LAGDHillClimber -- -P 1', 
                                                            '-Q \'weka.classifiers.bayes.net.search.local.LAGDHillClimber -- -P 1\''), axis=1)
    
    #dataframe['weka'] = dataframe.apply(lambda L: str(L.weka).replace('weka.classifiers.bayes.net.search.local.SimulatedAnnealing -- -U 10000', 
    #                                                        '\'weka.classifiers.bayes.net.search.local.SimulatedAnnealing -- -U 10000\''), axis=1)
    
    dataframe['weka'] = dataframe.apply(lambda L: str(L.weka).replace('-Q weka.classifiers.bayes.net.search.local.TabuSearch -- -P 1', 
                                                            '-Q \'weka.classifiers.bayes.net.search.local.TabuSearch -- -P 1\''), axis=1)
    

# plot_graph: plots histograms of standard deviations by dataset and considering all datasets
def plot_graph(series, path, title, x_label):
    series = series[~series.isna()]
    plt.hist(series)
    plt.title(title)
    plt.xlabel(x_label)
    plt.savefig(path)
    plt.show()
        

if __name__ == "__main__":

    if len(sys.argv) > 1:
        project = sys.argv[1]   
        dataset = sys.argv[2]  
        n_labels = sys.argv[3]            
        n_features = sys.argv[4]  
    else:
        project = '/home/dellvale/Testes/Cluster/GitHub/Experimento2/SurrogateModel/'
        dataset = 'flags-random' # nome+'-automl' ou nome+'-random'
        n_labels = 7           
        n_features = 19
    
    path_graphs = project + 'graphs/'
    path_metrics = project + 'metrics/'
    path_meta_datasets = project + 'meta_datasets/'
    
    df_unique = get_metrics(path_metrics, dataset)
        
    plot_graph(df_unique['std_fscore'], path_graphs+'graph-fscore-'+dataset+'.pdf', dataset, 'Std Fscore')
    plot_graph(df_unique['std_size'], path_graphs+'graph-size-'+dataset+'.pdf', dataset, 'Std Size')
    
    # selects the commands normalize, weka and meka, fscore and size
    df = df_unique[['normalize', 'meka', 'weka', 'median_fscore', 'median_size']]

    # load configuration file
    config = Configuration(n_features, n_labels)
    
    # gets the column indices of the new dataframe (dict_indexes) and the dictionary of indices (alg-hyp:index)
    list_indexes_cols, dict_indexes = get_all_indexes(config)
    
    result = []
    for i in range(df.shape[0]):
        # selects whether or not there is normalization and weka and meka commands for encoding
        is_norm = df['normalize'][i]
        meka = df['meka'][i] 
        weka = df['weka'][i]
        
        ind = codify(is_norm, meka, weka, dict_indexes, config)  
            
        # add individual to results
        if len(result) == 0:
            result = ind
        else:
            result = np.vstack((result, ind))
    
    # create encoded dataframe
    df_result = pd.DataFrame(result, columns=list_indexes_cols)
    df_result['F1 (macro averaged by label)'] = df['median_fscore']
    df_result['Model Size'] = df['median_size']
    
    print(dataset)
    print(df_unique.shape)
    print(df_result.shape)
    
    # save csv
    df_result.to_csv(path_meta_datasets+dataset+'.csv', index=False, sep=';') 