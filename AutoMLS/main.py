import sys

import os
import pathlib

import numpy as np

from configuration.Configuration import Configuration
from multiobjective.MLProblem import MLProblem
from multiobjective.MLSampling import MLSampling
from multiobjective.MLMutation import MLMutation
from multiobjective.IndividualUtils import IndividualUtils

from utils.Graphic import Graphic
from utils.ManipulateHistory import ManipulateHistory

from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.ux import UniformCrossover

from pymoo.core.termination import TerminateIfAny
from pymoo.core.duplicate import NoDuplicateElimination

from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.max_gen import MaximumGenerationTermination


def create_dir(destination, name_dir):    
    cmd = 'if [ ! -d '+destination+'/'+name_dir+' ]\nthen\n\tmkdir '+destination+'/'+name_dir+'\nfi'
    os.system(cmd)


if __name__ == "__main__":
    
    # parameters
    if len(sys.argv) > 1:
        n_labels = int(sys.argv[1]) 
        n_features = int(sys.argv[2])
        name_dataset = sys.argv[3]             
        is_dataset_sparse = bool(int(sys.argv[4]))
        java_command = sys.argv[5] 
        meka_classpath = sys.argv[6] 
        len_population = int(sys.argv[7])        
        number_generation = int(sys.argv[8])    
        k_folds = int(sys.argv[9])
        n_threads = int(sys.argv[10]) 
        limit_time = int(sys.argv[11]) 
        project = sys.argv[12]
        name_dir_res = sys.argv[13]
    else:
        n_labels = 6 
        n_features = 72 
        name_dataset = 'emotions' 
        is_dataset_sparse = False
        java_command = '/home/dellvale/miniconda3/envs/AmbienteMEKA/bin/java'
        meka_classpath = '/home/dellvale/scikit_ml_learn_data/meka/meka-release-1.9.2/lib/'
        len_population = 8
        number_generation = 10
        k_folds = 3
        n_threads = 4
        limit_time = 60 # seconds
        project = '/home/dellvale/Testes/Cluster/GitHub/Experimento2/AutoMLS'
        name_dir_res = 'exe1'
     
    n_gene = 28 # individual size
    termination_period = 10 # number of generations without changes
    termination_tol = 0.001 # improvement tolerance
    
    # log
    log_file = project+'/log/'+'log_'+name_dataset+'_'+name_dir_res+'.txt'
    
    # results folder
    create_dir(project+'/results/'+name_dataset, name_dir_res)
    path_res = project+'/results/'+name_dataset+'/'+name_dir_res
    metrics_file = path_res+'/metrics.csv'
    
    # feature,label
    config = Configuration(n_features, n_labels)
    path_dataset = project+'/datasets/'+name_dataset
    problem = MLProblem(config, java_command, meka_classpath, k_folds, n_threads, limit_time, name_dataset, path_dataset, is_dataset_sparse, n_labels, n_features, log_file, metrics_file)
    
    algorithm = NSGA2(
        pop_size=len_population,
        sampling=MLSampling(config, n_gene, is_dataset_sparse),
        crossover=UniformCrossover(prob=0.5),
        mutation=MLMutation(0.05, config),
        eliminate_duplicates=NoDuplicateElimination() 
    )
    
    # termination: maximum number of generations or tolerance of 'tol' for 'period' generations
    termination = TerminateIfAny(MaximumGenerationTermination(number_generation), RobustTermination(MultiObjectiveSpaceTermination(tol=termination_tol, n_skip=0), period=termination_period))
    
    res = minimize(
        problem,
        algorithm,
        termination,
        save_history=True,
        verbose=True
    )    
    
    # Results --------------------------------------------------------------
    
    # Graph - objective space
    Graphic.plot_scatter(res.F[:, 0], res.F[:, 1], 'Objective Space', '-F1', 'Model Size', path_res+'/ObjectiveSpace.png')
    
    # Graph - hypervolume
    ref_point = np.array([0, 1e9])
    n_evals, hist_F, hv = ManipulateHistory.get_hypervolume(res, ref_point)
    Graphic.plot_graphic(n_evals, hv, 'Convergence-Hypervolume', 'Evaluations', 'Hypervolume', path_res+'/Hypervolume.png')    
    
    # Prepare results for saving to file
    output_data = f'Number of queries on the classifier and objectives map: {problem.rep}\n'
    
    output_data += f'Number of classifiers exceeding the time limit: {problem.classifier_limit_time}\n'
    
    output_data += f'Number exception: {problem.classifier_exception}\n'

    output_data += f'Execution time:{res.exec_time}\n'
      
    output_data += f'Best solution found:\n'
    for individual in res.X:
        output_data += f'{individual}\n'
     
    output_data += 'Classifiers:\n'
    for individual in res.X:
        is_normalize, meka_command, weka_command = IndividualUtils.get_commands(config, individual)
        output_data += f"Normalize:{is_normalize}\n{meka_command}\n{weka_command}\n"
        
    output_data += f"Function value:\n{res.F}\n"
    
    list_f1 = [] # -f1
    list_f2 = [] # model size
    for l in res.F:
        list_f1 = np.append(list_f1, l[0])
        list_f2= np.append(list_f2, l[1])
    
    # f1
    output_data += 'F1 (macro averaged by label)\n['
    for i in range(len(list_f1)-1):
        output_data += str(-list_f1[i])+','
    output_data += str(-list_f1[len(list_f1)-1])+']\n'
     
    # size
    output_data += 'Model size\n['
    for i in range(len(list_f2)-1):
        output_data += str(list_f2[i])+','
    output_data += str(list_f2[len(list_f2)-1])+']\n'
    
    # Evaluation
    output_data += 'Evaluation:\n['
    for i in range(len(n_evals)-1):
        output_data += str(n_evals[i])+','
    output_data += str(n_evals[len(n_evals)-1])+']\n'
    
    # Hypervolume
    output_data += 'Hypervolume:\n['
    for i in range(len(hv)-1):
        output_data += str(hv[i])+','
    output_data += str(hv[len(hv)-1])+']\n'
    
    # History
    output_data += 'History:\n'
    for array in hist_F:
        if len(array) == 1:
            output_data += str(array[0][0])+', '+str(array[0][len(array[0])-1])+'\n'
        else:
            for i in range(len(array)-1):
                output_data += str(array[i][0])+', '+str(array[i][len(array[i])-1])+', '
            output_data += str(array[i+1][0])+', '+str(array[i+1][len(array[i+1])-1])+'\n'
    
    # save results
    output_path = pathlib.Path(f'{path_res}/results.txt')
    output_path.write_text(output_data)
    
    print(output_data)