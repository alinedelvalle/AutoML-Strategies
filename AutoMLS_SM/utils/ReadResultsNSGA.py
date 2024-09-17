''' 
Estrutra do arquivo:
    
Execution time:xxx
Best solution found:
[ x x x ]
...
[ x x x ]
Classifiers:
Classifier 1
...
ClassifierN
Function value:
[[x  x]
 ...
 [x   x]
F_score
[x,...,x]
Time
[x,...,x]
Hypervolume:
[x,...,x]
Evaluation:
[x,...,x]
History:
f_score, time, ..., f_score, time
f_score, time, ..., f_score, time
f_score, time, ..., f_score, time
    
'''

import numpy as np


class ReadResultsNSGA:

    def read_fscore(file_name):
        # lê arquivo de resultados
        file_results = open(file_name, 'r')    
           
        reading_fscore = False
        list_fscore = []
        while file_results:
            line = file_results.readline()
            
            if line == '':
                break

            if 'F_score' in line:
                reading_fscore = True
                continue
                
            if reading_fscore == True:           
                line = line.replace('[', '')
                line = line.replace(']\n', '')
                array = line.split(',')
                
                for s in array:
                    list_fscore = np.append(list_fscore, float(s))

                break
            
        return list_fscore


    def read_time(file_name):
        # lê arquivo de resultados
        file_results = open(file_name, 'r')    
           
        reading_time = False
        list_time = []
        while file_results:
            line = file_results.readline()
            
            if line == '':
                break

            if 'Time' in line:
                reading_time = True
                continue
                
            if reading_time == True:            
                line = line.replace('[', '')
                line = line.replace(']\n', '')
                array = line.split(',')
                
                for s in array:
                    list_time = np.append(list_time, float(s))

                break
            
        return list_time
    
    
    def read_classifiers(file_name):
        # lê arquivo de resultados
        file_results = open(file_name, 'r')    
           
        reading_classifier = False
        list_classifiers = []
        while file_results:
            line = file_results.readline()
            
            if line == '':
                break

            if 'Function value' in line:
                break;
                
            if 'Classifiers' in line:
                reading_classifier = True
                continue
                
            if reading_classifier == True:  
                line = line.replace('["', '')
                line = line.replace('\n"', '')
                line = line.replace(']', '')
                list_classifiers.append(line)
            
        return list_classifiers


    '''
    def read_hypervolume(file_name):
        # lê arquivo de resultados
        file_results = open(file_name, 'r')    
           
        reading_hypervolume = False
        list_hypervolume = []
        while file_results:
            line = file_results.readline()
            
            if line == '':
                break

            if 'Hypervolume' in line:
                reading_hypervolume = True
                continue
                
            if reading_hypervolume == True:            
                line = line.replace('[', '')
                line = line.replace(']\n', '')
                array = line.split(',')
                
                for s in array:
                    list_hypervolume = np.append(list_hypervolume, float(s))

                break
            
        return list_hypervolume


    def read_n_evals(file_name):
        # lê arquivo de resultados
        file_results = open(file_name, 'r')    
           
        reading_n_evals = False
        list_n_evals = np.array([], dtype=int)
        while file_results:
            line = file_results.readline()
            
            if line == '':
                break

            if 'Evaluation' in line:
                reading_n_evals = True
                continue
                
            if reading_n_evals == True:
                line = line.replace('[', '')
                line = line.replace(']', '')
                array = line.split(',')
                
                for s in array:
                    list_n_evals = np.append(list_n_evals, int(s))

                break
            
        return list_n_evals
    '''


    def read_hist(file_name):
        # lê arquivo de resultados
        file_results = open(file_name, 'r')    
           
        reading_hist = False
        list_hist = []
        while file_results:
            line = file_results.readline()
            
            if line == '':
                break

            if 'History' in line:
                reading_hist = True
                continue
                
            if reading_hist == True:
                line = line.replace(' ', '')
                array = line.split(',')
                
                list_aux = []
                for i in range(0, len(array), 2):  
                    v = [float(array[i]), float(array[i+1])]
                    list_aux.append(v)
                 
                list_aux = np.asanyarray(list_aux)
                list_hist.append(list_aux)

                continue
        
        return list_hist