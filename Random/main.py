import os
import sys
import pathlib

import time

import numpy as np

from pymoo.indicators.hv import HV

from configuration.Configuration import Configuration

from multiobjective.RandomSearch import RandomSearch
from utils.AlgorithmsHyperparameters import AlgorithmsHyperparameters
from multiobjective.Pareto_Froint import FNDS


def create_dir(destination, name_dir):    
    cmd = 'if [ ! -d '+destination+'/'+name_dir+' ]\nthen\n\tmkdir '+destination+'/'+name_dir+'\nfi'
    os.system(cmd)


if __name__ == '__main__':
    
    # se há parâmetros
    if len(sys.argv) > 1:
        n_labels = int(sys.argv[1]) 
        n_features = int(sys.argv[2]) 
        name_dataset = sys.argv[3]    
        is_dataset_sparse = bool(int(sys.argv[4]))          
        java_command = sys.argv[5] 
        meka_classpath = sys.argv[6] 
        n_sample = int(sys.argv[7])  
        kfolds = int(sys.argv[8])
        n_threads = int(sys.argv[9]) 
        limit_time = int(sys.argv[10]) 
        project = sys.argv[11]
        name_dir_res = sys.argv[12] 
    else:
        n_labels = 6 
        n_features = 72 
        name_dataset = 'emotions' 
        is_dataset_sparse = False
        java_command = '/' # path para o java do environment
        meka_classpath = '' # path path to the lib folder of the MEKA library
        n_sample = 20
        kfolds = 3
        n_threads = 4
        limit_time = 60 # seconds
        project = '' # project path
        name_dir_res = 'exe1' # folder name in folder results

        
    config = Configuration(n_features, n_labels)
    
    path_dataset = project+'/datasets/'+name_dataset
    log_file = project+'/log/'+'log_'+name_dataset+'-'+name_dir_res+'.txt'
    
    # results folder
    create_dir(project+'/results/'+name_dataset, name_dir_res)
    path_res = project+'/results/'+name_dataset+'/'+name_dir_res
    metrics_file = path_res+'/metrics.csv'
        
    rs = RandomSearch(config, java_command, meka_classpath, kfolds, n_threads, limit_time, name_dataset, path_dataset, is_dataset_sparse, n_labels, n_features, log_file, metrics_file)
    
    start_time = time.time()
    list_points, pareto_froint = rs.evaluate(n_sample)
    end_time = time.time()
    
    # Results --------------------------------------------------------------
    AlgorithmsHyperparameters.to_file(name_dataset, metrics_file)
    
    output_data = f'Number of queries on the classifier and objectives map: {rs.rep}\n'
    
    output_data += f'Number of classifiers exceeding the time limit: {rs.classifier_limit_time}\n'

    output_data += f'Execution time: {end_time - start_time}\n'
    
    output_data += '\nClassifiers:\n'
    for pto in pareto_froint:
        output_data += str(pto.norm) +'\n'
        output_data += pto.meka +'\n'
        if pto.weka is not None:
            output_data += pto.weka +'\n\n'
        else:
            output_data += '\n'
     
    output_data += 'Objective: '    
    for pto in pareto_froint:
        output_data += f'{pto.obj1}, {pto.obj2}\n'
    output_data += '\n'    
     
    list_obj1 = []
    output_data += '\nF1 (macro averaged by label)\n'
    for pto in pareto_froint:
        output_data += f'{-pto.obj1} '
        list_obj1.append(pto.obj1)
    output_data += '\n'

    list_obj2 = []
    output_data += '\nModel Size\n'
    for pto in pareto_froint:
        output_data += f'{pto.obj2} '
        list_obj2.append(pto.obj2)
    output_data += '\n'
    
    output_data += '\nHypervolume:\n'
    F = np.array([list_obj1, list_obj2]).transpose()    
    ind = HV(ref_point=np.array([0, 1e9]))
    hv = ind(F)
    output_data += f'{hv}'
    
    # Save results
    output_path = pathlib.Path(f'{path_res}/results.txt')
    output_path.write_text(output_data)
    
    # Graph - objective space
    FNDS().plot_froint(list_points, pareto_froint, 'Objective Space', '-F1', 'Model Size', path_res+'/ObjectiveSpace')
    
    print(output_data)