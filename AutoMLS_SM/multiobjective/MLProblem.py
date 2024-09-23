import numpy as np

import  joblib

from pymoo.core.problem import Problem

from multiobjective.IndividualUtils import IndividualUtils

from utils.AlgorithmsHyperparameters import AlgorithmsHyperparameters

from multiobjective.MetaDatasetUtils import *

from multiprocessing.pool import ThreadPool 


class MLProblem(Problem):
    
    def __init__(self, name_dataset, config, n_threads, path_model_f1, path_model_size, file_metrics):
        super().__init__(n_var=28, n_obj=2) 
        
        self.name_dataset = name_dataset
        self.config = config
        self.n_threads = n_threads
        self.path_model1 = path_model_f1
        self.path_model2 = path_model_size
        self.file_metrics = file_metrics
        
        self.n_ger = 0
        self.list_indexes_cols, self.dict_indexes = get_all_indexes(self.config)
        
    
    def my_eval(self, param):
        cmd, model1, model2 = param
        is_normalize, meka_command, weka_command = cmd
        
        # get the encoded algorithm and a test dataframe   
        alg_cod = codify(is_normalize, meka_command, weka_command, self.dict_indexes, self.config) 
        x_test = pd.DataFrame([alg_cod], columns=self.list_indexes_cols)
        x_test = x_test + 1
        x_test.fillna(0, inplace=True)
        
        # F1
        f1 =  model1.predict(x_test)[0]
        
        # Model size
        log_model_size = model2.predict(x_test)
        model_size = np.exp(log_model_size)[0]
        
        AlgorithmsHyperparameters.add_metrics(is_normalize, meka_command, weka_command, f1, model_size)
        
        return [-f1, model_size]
    
        
    def _evaluate(self, X, out, *args, **kwargs):   
        pool = ThreadPool(self.n_threads)
        
        models_f1 = [joblib.load(self.path_model1) for k in range(self.n_threads)]
        models_f2 = [joblib.load(self.path_model2) for k in range(self.n_threads)]
        
        # prepare the parameters for the pool
        # converts numeric vectors into MLC algorithms
        params = [[(IndividualUtils.get_commands(self.config, X[k]), models_f1[k%self.n_threads], models_f2[k%self.n_threads])] for k in range(len(X))]
        
        # pool de threads
        F = pool.starmap(self.my_eval, params)
        
        pool.close()  
        
        # store the function values and return them.
        out["F"] = np.array(F, dtype=object)
        
        AlgorithmsHyperparameters.to_file(self.name_dataset, self.file_metrics)