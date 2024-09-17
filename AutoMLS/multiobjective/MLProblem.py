import numpy as np

from pymoo.core.problem import Problem

from multiobjective.IndividualUtils import IndividualUtils

from utils.AlgorithmsHyperparameters import AlgorithmsHyperparameters

# meka_adapted: com timeout e tamanho do classificador
from meka.meka_adapted import MekaAdapted

from skmultilearn.dataset import load_from_arff

from multiprocessing.pool import ThreadPool 
from subprocess import TimeoutExpired


class MLProblem(Problem):
    
    def __init__(self, config, java_command, meka_classpath, k_folds, n_threads, limit_time, name_dataset, path_dataset, is_dataset_sparse, n_labels, n_features, log_file, metrics_file):
        super().__init__(n_var=28, n_obj=2) 

        self.config = config
        self.java_command = java_command
        self.meka_classpath = meka_classpath        
        self.n_threads = n_threads
        self.limit_time = limit_time
        
        self.name_dataset = name_dataset
        self.path_dataset = path_dataset
        self.is_dataset_sparse = is_dataset_sparse
        self.n_labels = n_labels
        self.n_features = n_features
        
        self.kfolds = k_folds
        self.n_example_test = [self.__get_n_example_test(self.path_dataset+'/'+self.name_dataset+'-test-'+str(k)+'.arff') for k in range(self.kfolds)]
        self.n_example_test_norm = [self.__get_n_example_test(self.path_dataset+'/'+self.name_dataset+'-norm-test-'+str(k)+'.arff') for k in range(self.kfolds)] if self.is_dataset_sparse == False else []
        
        self.map_algs_objs = {} # algorithm: objective1, objective2
        self.rep = 0 # number of times the classifier was reused from the map
        self.classifier_limit_time = 0 # number of times the classifier timed out
        self.classifier_exception = 0
        
        self.log_file = log_file
        self.n_ger = 0
        self.metrics_file = metrics_file
    
    
    def to_file_ger(self):
        file = open(self.log_file, 'a')
        file.write(f'Generation: {str(self.n_ger)}\n')
        file.close()
        
        
    def __get_n_example_test(self, name_dataset_test):
        # extrair do dataset: features e test example
        x_test, y_test = load_from_arff(
            name_dataset_test, 
            label_count=self.n_labels,
            label_location='end',
            load_sparse=False,
            return_attribute_definitions=False
        )
        
        n_example_test = x_test.shape[0]
        
        return n_example_test
    
    
    def my_eval(self, param):
        print('Starting ...')
        
        flag = False # there are no runtime errors (exception ou timeout)
        is_normalize, meka_command, weka_command = param
        command = str(is_normalize) + ' ' + meka_command
        
        if weka_command is not None:
            command = command + ' -W ' + weka_command
            
        # query the objective map for the algorithm
        # if the algorithm has already been evaluated on k folds
        # use the average values ​​of the objectives already calculated
        if command in self.map_algs_objs.keys():   
            f1 = self.map_algs_objs.get(command)[0] 
            f2 = self.map_algs_objs.get(command)[1]
            self.rep += 1  
        else:
            # prepare meka command
            meka = MekaAdapted(
                meka_classifier = meka_command,
                weka_classifier = weka_command,
                meka_classpath = self.meka_classpath, 
                java_command = self.java_command,
                timeout = self.limit_time
            )
            
            # get dataset name
            if is_normalize == False:
                train_data = self.path_dataset+'/'+self.name_dataset+'-train-'
                test_data = self.path_dataset+'/'+self.name_dataset+'-test-'
            else:
                train_data = self.path_dataset+'/'+self.name_dataset+'-norm-train-'
                test_data = self.path_dataset+'/'+self.name_dataset+'-norm-test-'
            
            # stores objective values ​​for k-folds
            list_f1 = np.array([], dtype=float)
            list_f2 = np.array([], dtype=float)
            
            # runs the algorithm for k-folds
            for k in range(self.kfolds): 
                model_size = 0
                try:
                    train_data_k = train_data+str(k)+'.arff'
                    test_data_k = test_data+str(k)+'.arff'
                    n_test_example = self.n_example_test_norm[k] if is_normalize else self.n_example_test[k]                    
                    
                    # predictions
                    meka.fit_predict(n_test_example, self.config.n_labels, train_data_k, test_data_k)
                    
                    statistics = meka.statistics
                    model_size = meka.len_model_file 
                    f1 = statistics.get('F1 (macro averaged by label)')
                    f2 = model_size
                    
                    list_f1 = np.append(list_f1, f1)
                    list_f2 = np.append(list_f2, f2)
                    
                except TimeoutExpired:
                    flag = True
                    statistics = {}
                    self.classifier_limit_time += 1
                    print(f'---------- TimeoutExpired ----------\n{k} {command}\n---------- TimeoutExpired ----------')
                    
                except Exception as e:
                    flag = True
                    statistics = {}
                    self.classifier_exception += 1
                    print(f'---------- Exception ----------\n{e}\n{k} {command}\n---------- Exception ----------')
                
                AlgorithmsHyperparameters.add_metrics(k, is_normalize, meka_command, weka_command, statistics, model_size)
                
            # não houve erro    
            if flag == False or len(list_f1) > 0:
                f1 = list_f1.mean()
                f2 = list_f2.mean()
            else:    
                f1 = 0 # f1
                f2 = 1e9 # size (1GB)
                
            self.map_algs_objs[command] = (f1, f2)  
            
            print(f'F1: {list_f1} {f1}\nF2: {list_f2} {f2}')
        
        return [-f1, f2]
    
        
    def _evaluate(self, X, out, *args, **kwargs):   
        pool = ThreadPool(self.n_threads)
        
        # prepare the parameters for the pool
        params = [[IndividualUtils.get_commands(self.config, X[k])] for k in range(len(X))]
        
        # pool de threads
        F = pool.starmap(self.my_eval, params)
        
        pool.close()  
        
        # store the function values and return them.
        out["F"] = np.array(F, dtype=object)
        
        # log
        self.n_ger += 1 
        self.to_file_ger() # stores the current generation
        
        AlgorithmsHyperparameters.to_file(self.name_dataset, self.metrics_file)