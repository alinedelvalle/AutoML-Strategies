import pandas as pd
import numpy as np

class AlgorithmsHyperparameters :
    
    list_metrics = []
    id_general = 1
    flag = True
    
    title = ['id', 'normalize', 'meka', 'weka', 'F1 (macro averaged by label)', 'Model Size']
    
    # statistics is a dict
    def add_metrics(is_normalize, meka_command, weka_command, f1, model_size):   
        metrics = []
        
        metrics.append(AlgorithmsHyperparameters.id_general)
        metrics.append(is_normalize)
        metrics.append(meka_command)
        metrics.append(weka_command)
        metrics.append(f1)
        metrics.append(model_size)
            
        AlgorithmsHyperparameters.id_general = AlgorithmsHyperparameters.id_general + 1
        AlgorithmsHyperparameters.list_metrics.append(metrics)
        
        
    def to_file(dataset_name, file_name):        
        df = pd.DataFrame(data = AlgorithmsHyperparameters.list_metrics, columns=AlgorithmsHyperparameters.title)
            
        list_dataset = np.full((len(AlgorithmsHyperparameters.list_metrics)), dataset_name)
        
        df.insert(1, 'Dataset', list_dataset)
        
        if AlgorithmsHyperparameters.flag == True:
            df.to_csv(file_name, sep=';', index=False, mode='a')
        else:
            df.to_csv(file_name, sep=';', index=False, header=False, mode='a')
        
        AlgorithmsHyperparameters.list_metrics = []
        AlgorithmsHyperparameters.flag = False