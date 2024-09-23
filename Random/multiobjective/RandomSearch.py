import numpy as np

from multiprocessing.pool import ThreadPool 
from subprocess import TimeoutExpired

from skmultilearn.dataset import load_from_arff

from multiobjective.Sampling import Sampling
from multiobjective.CommandMW import CommandMW
from meka.meka_adapted import MekaAdapted
from utils.AlgorithmsHyperparameters import AlgorithmsHyperparameters
from multiobjective.Pareto_Froint import Point, FNDS


class RandomSearch():
    
    def __init__(self, config, java_command, meka_classpath, kfolds, n_threads, limit_time, name_dataset, path_dataset, is_dataset_sparse, n_labels, n_features, log_file, metrics_file):
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
        self.log_file = log_file
        self.metrics_file = metrics_file
        
        self.kfolds = kfolds
        self.n_example_test = [self.__get_n_example_test(self.path_dataset+'/'+self.name_dataset+'-test-'+str(k)+'.arff') for k in range(self.kfolds)]
        self.n_example_test_norm = [self.__get_n_example_test(self.path_dataset+'/'+self.name_dataset+'-norm-test-'+str(k)+'.arff') for k in range(self.kfolds)] if self.is_dataset_sparse == False else []
        
        self.map_algs_objs = {} # algorithm dictionary: objective1, objective2
        self.rep = 0 # number of times the classifier was reused from the map
        self.classifier_limit_time = 0 # number of times the classifier timed out
        self.classifier_exception = 0 # number of exceptions
        
        self.list_points = []
        self.n = 0
        

    def to_file(self, n):
        file = open(self.log_file, 'a')
        file.write(f'{str(n)}\n')
        file.close()
        
        
    def __get_n_example_test(self, name_dataset_test):
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
        is_normalize, meka_command, weka_command = param
        
        # command string
        command = str(is_normalize) + ' ' + meka_command
        
        if weka_command is not None:
            command = command + ' -W ' + weka_command
        
        if is_normalize == True:
            print(f'\n---\n{is_normalize}, {meka_command}, {weka_command}\n---\n')
        
        # query the objective map for algorithms
        # if the algorithm has already been evaluated in k folds
        # use the average values ​​of the objectives already calculated
        if command in self.map_algs_objs.keys():   
            f1 = self.map_algs_objs.get(command)[0] 
            f2 = self.map_algs_objs.get(command)[1]
            self.rep += 1  
        else:
            print('Started ...\n')
            
            # meka command
            meka = MekaAdapted(
                meka_classifier = meka_command,
                weka_classifier = weka_command,
                meka_classpath = self.meka_classpath, 
                java_command = self.java_command,
                timeout = self.limit_time
            )
            
            # dataset names
            if is_normalize == False:
                train_data = self.path_dataset+'/'+self.name_dataset+'-train-'
                test_data = self.path_dataset+'/'+self.name_dataset+'-test-'
            else:
                train_data = self.path_dataset+'/'+self.name_dataset+'-norm-train-'
                test_data = self.path_dataset+'/'+self.name_dataset+'-norm-test-'
            
            list_f1 = np.array([], dtype=float)
            list_f2 = np.array([], dtype=float)
            
            # runs the algorithm for k-folds
            for k in range(self.kfolds):  
                model_size = None
                try:
                    train_data_k = train_data+str(k)+'.arff'
                    test_data_k = test_data+str(k)+'.arff'
                    n_test_example = self.n_example_test_norm[k] if is_normalize else self.n_example_test[k]                   
                    
                    # predictions
                    meka.fit_predict(n_test_example, self.n_labels, train_data_k, test_data_k)
                    
                    statistics = meka.statistics
                    model_size = meka.len_model_file 
                    f1 = statistics.get('F1 (macro averaged by label)')
                    f2 = model_size
                    
                    list_f1 = np.append(list_f1, f1)
                    list_f2 = np.append(list_f2, f2)
                    
                except TimeoutExpired:
                    self.classifier_limit_time += 1
                    statistics = {}
                    print(f'---------- TimeoutExpired ----------\n{k, is_normalize, meka_command, weka_command}\n---------- TimeoutExpired ----------')
                        
                except Exception as e:
                    self.classifier_exception += 1
                    statistics = {}
                    print(f'---------- Exception ----------\n{e}\n{k, is_normalize, meka_command, weka_command}\n---------- Exception ----------')
            
                AlgorithmsHyperparameters.add_metrics(k, is_normalize, meka_command, weka_command, statistics, model_size)
            
            if len(list_f1) > 0:
                f1 = np.mean(list_f1)
                f2 = np.mean(list_f2)
            else:
                f1 = 0 # f1
                f2 = 1e9 # model size
                    
            self.map_algs_objs[command] = (f1, f2)  
            
            print(f'F1: {list_f1} {f1}\nF2: {list_f2} {f2}')
        
        # creates a point on the pareto frontier and adds it to the list of points
        pto = Point(-f1, f2, is_normalize, meka_command, weka_command)
        self.list_points.append(pto)
        
        self.n += 1
        
        if self.n%10==0:
            self.to_file(self.n) # log

    
    
    def evaluate(self, n_sample):
        pool = ThreadPool(self.n_threads)
        
        # sampling
        sample = Sampling(self.config, self.is_dataset_sparse) 
        
        # prepare the parameters for the pool
        # converts numeric vectors into MLC algorithms
        params = [[CommandMW.get_commands(self.config, sample.do())] for i in range(n_sample)]
        
        # pool de threads
        pool.starmap(self.my_eval, params)
        
        pool.close() 
        
        # Pareto frontier
        fnds = FNDS()
        pareto_froint = fnds.execute(self.list_points) 
        
        # return list of points and Pareto frontier
        return self.list_points, pareto_froint