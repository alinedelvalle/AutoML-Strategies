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
from utils.AlgorithmsHyperparameters import AlgorithmsHyperparameters

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
    
    # se há parâmetros
    if len(sys.argv) > 1:
        n_labels = int(sys.argv[1]) 
        n_features = int(sys.argv[2])
        name_dataset = sys.argv[3]            
        is_dataset_sparse = bool(int(sys.argv[4]))
        len_population = int(sys.argv[5])        
        number_generation = int(sys.argv[6])   
        n_threads = int(sys.argv[7]) 
        project = sys.argv[8]
        name_dir_res = sys.argv[9]
    else:        
        n_labels = 6
        n_features = 72
        name_dataset = 'emotions' 
        is_dataset_sparse = False
        len_population = 10
        number_generation = 10
        n_threads = 4
        project = '' # project path
        name_dir_res = 'exe1' # folder name in folder results
        
     
    n_gene = 28 # do indivíduo
    termination_period = 10
    termination_tol = 0.001 # de melhora
    
    # results folder
    create_dir(project+'/results/'+name_dataset, name_dir_res)
    path_res = project+'/results/'+name_dataset+'/'+name_dir_res
    
    # metrics
    file_metrics = path_res+'/predict_metrics.csv'
    
    # feature,label
    config = Configuration(n_features, n_labels)

    # models 
    path_models = project+'/models'
    path_model1 = path_models+'/model-'+name_dataset+'-k3-obj1.sav'
    path_model2 = path_models+'/model-'+name_dataset+'-k3-obj2.sav'
    
    problem = MLProblem(name_dataset,
                        config, 
                        n_threads, 
                        path_model1,
                        path_model2,
                        file_metrics)
    
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
    
    AlgorithmsHyperparameters.to_file(name_dataset, file_metrics)
    
    # Graph - objective space
    Graphic.plot_scatter(res.F[:, 0], res.F[:, 1], 'Objective Space', '-F1', 'Model Size', path_res+'/ObjectiveSpace.png')
    
    # Graph - hypervolume
    ref_point = np.array([0, 1e9])
    n_evals, hist_F, hv = ManipulateHistory.get_hypervolume(res, ref_point)
    Graphic.plot_graphic(n_evals, hv, 'Convergence-Hypervolume', 'Evaluations', 'Hypervolume', path_res+'/Hypervolume.png')    
    
    output_data = f'Execution time:{res.exec_time}\n'
      
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