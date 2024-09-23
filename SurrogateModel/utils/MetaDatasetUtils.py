import re

import inspect

import numpy as np

import shlex

import pandas as pd

from configuration.Configuration import Configuration


def get_all_indexes(config):
    dict_indexes = {}
    
    list_indexes_cols = np.array(['normalize'])
    
    dict_indexes['normalize'] = 0
    index, list_indexes_cols, dict_indexes = get_indexes(config.get_ml_ensemble_config(), 1, list_indexes_cols, dict_indexes)
    index, list_indexes_cols, dict_indexes = get_indexes(config.get_ml_config(), index, list_indexes_cols, dict_indexes)
    index, list_indexes_cols, dict_indexes = get_indexes(config.get_sl_ensemble_config(), index, list_indexes_cols, dict_indexes)
    index, list_indexes_cols, dict_indexes = get_indexes(config.get_sl_config(), index, list_indexes_cols, dict_indexes, True)
    index, list_indexes_cols, dict_indexes = get_indexes(config.get_sl_kernel_config(), index, list_indexes_cols, dict_indexes)
    
    return list_indexes_cols, dict_indexes


# Gets the dataframe/dataset indices for ensemble, SLC and MLC configurations
def get_indexes(config, i, list_indexes, dict_indexes, is_scl=False):
    for algorithm, hyperparameters in config.items():
        #algorithm = algorithm.replace(without_pattern, '')
        list_indexes = np.append(list_indexes, algorithm)
        dict_indexes[algorithm] = i
        i = i + 1

        for hyp, values in hyperparameters.items():
            if hyp == '-normalize':
                continue
            elif hyp == 'if':
                function = values
                funcString = str(inspect.getsourcelines(function)[0])
                funcString = funcString.split('\'if\'')[1].split('if')
                blockThen = funcString[0]
                blockElse = funcString[1].split('else')[1]
                list_function = np.append(re.findall('-\D', blockThen), re.findall('-\D', blockElse))
                list_function = np.unique(list_function)
                for h in list_function: # for each hyperparameter of the function
                    list_indexes = np.append(list_indexes, algorithm+h)
                    dict_indexes[algorithm+h] = i
                    i = i + 1
            elif hyp != '-W' or is_scl == True:
                list_indexes = np.append(list_indexes, algorithm+hyp)
                dict_indexes[algorithm+hyp] = i
                i = i + 1

    return i, list_indexes, dict_indexes


# checks if a string represents a number
def isNumber(n):
  try:
      float(n)
  except ValueError:
      return False
  return True


# encodes meka and weka commands into an extended individual
def codify(norm, meka_cmd, weka_cmd, dict_indexes, config):
    # creates individual vector
    individual = np.full(len(dict_indexes), np.nan) # -1
    
    is_pt = False
    is_mlc_ensemble = False
    
    individual[0] = 0 if norm == False else 1
    
    # converts meka and weka commands into arrays
    meka_cmd = shlex.split(meka_cmd)
    
    if not pd.isna(weka_cmd):
        weka_cmd = shlex.split(weka_cmd) 
    
    # get multi-label classification algorithm
    algorithm = meka_cmd[0]
    index_cmd = 1
    
    # set as 1 the index of the individual referring to the algorithm
    index_individual = dict_indexes.get(algorithm)
    individual[index_individual] = 0 # 1
    
    # Ensemble MLC -----------------------------------------------------------
    if 'meta' in algorithm:
        is_mlc_ensemble = True
        config_ml = config.get_ml_ensemble_config()
        config_algorithm = config_ml.get(algorithm)
        
        params = {}
        for key, values in config_algorithm.items():
            
            if key == '-W': 
                # get MLC algorithm
                index_cmd = index_cmd + 1 # -W
                algorithm = meka_cmd[index_cmd]
                index_cmd = index_cmd + 1

                # get the algotithm configuration
                config_ml = values
                config_algorithm = config_ml.get(algorithm)

                # sets as 1 index of the individual referring to the algorithm
                index_individual = dict_indexes.get(algorithm)
                individual[index_individual] = 0 # 1

            elif key == 'if':
                function = config_algorithm[key]
                return_function = function(params)
                
                if isinstance(return_function, dict):
                    for key_dict, values_dict in return_function.items():
                        # get the individual's index
                        index_individual = dict_indexes.get(algorithm+key_dict)
                        index_cmd, individual, params = get_one_cod(key_dict, values_dict, meka_cmd, index_cmd, individual, index_individual, params) 
            
            else:
                # sets the individual's index to the hyperparameter value
                index_individual = dict_indexes.get(algorithm+key)
                index_cmd, individual, params = get_one_cod(key, values, meka_cmd, index_cmd, individual, index_individual, params)            
        
        # --
        index_cmd = index_cmd + 1 
    
    # MLC ---------------------------------------------------------------------  
    # If it is not an MLC ensemble, get the algorithm configuration      
    if is_mlc_ensemble == False:
        config_ml = config.get_ml_config() 
        config_algorithm = config_ml.get(algorithm)
     
    params = {}
    for key, values in config_algorithm.items():
        if key == '-normalize':
            continue
        elif key == '-W': 
            # the multi-label algorithm is from the problem transformation approach
            is_pt = True
            # get the SLC algorithms settings
            config_slc = values

        elif key == 'if':
            function = config_algorithm[key]
            return_function = function(params)
            
            if isinstance(return_function, dict):
                for key_dict, values_dict in return_function.items():
                    # get the individual's index
                    index_individual = dict_indexes.get(algorithm+key_dict)
                    index_cmd, individual, params = get_one_cod(key_dict, values_dict, meka_cmd, index_cmd, individual, index_individual, params) 
        else:
            # sets the individual's index to the hyperparameter value
            index_individual = dict_indexes.get(algorithm+key)
            index_cmd, individual, params = get_one_cod(key, values, meka_cmd, index_cmd, individual, index_individual, params)            
    
    # SLC Ensemble ou SLC -----------------------------------------------------
    if is_pt == True:
        # get single-label classification algorithm (SLC or SLC ensemble)
        algorithm = weka_cmd[0]
        index_cmd = 1
        # --
        index_cmd = index_cmd + 1

        # get the configuration of the single-label classification algorithm
        config_algorithm = config_slc.get(algorithm)

        # sets as 1 the index of the individual referring to the algorithm
        index_individual = dict_indexes.get(algorithm)
        individual[index_individual] = 0 # 1
        
        # ensemble de SLC -----------------------------------------------------
        if 'meta' in algorithm or 'LWL' in algorithm:
            
            params = {}
            for key, values in config_algorithm.items():
                
                # get the base SLC algorithm information
                if key == '-W': 
                    index_cmd = index_cmd + 1 # -w
                    # get the base SLC algorithm
                    algorithm = weka_cmd[index_cmd]
                    index_cmd = index_cmd + 1
                    # --
                    index_cmd = index_cmd + 1
                    
                    # sets as 1 the index of the individual referring to the SLC algorithm
                    index_individual = dict_indexes.get(algorithm)
                    individual[index_individual] = 0 # 1

                    # get the SLC algorithm configuration
                    config_slc = values
                    config_algorithm = config_slc.get(algorithm)

                elif key == 'if':
                    function = config_algorithm[key]
                    return_function = function(params)

                    if isinstance(return_function, dict):
                        for key_dict, values_dict in return_function.items():
                            # obtains the individual's index
                            index_individual = dict_indexes.get(algorithm+key_dict)
                            index_cmd, individual, params = get_one_cod(key_dict, values_dict, weka_cmd, index_cmd, individual, index_individual, params) 
                else:
                    # sets the individual's index to the hyperparameter value
                    index_individual = dict_indexes.get(algorithm+key)
                    index_cmd, individual, params = get_one_cod(key, values, weka_cmd, index_cmd, individual, index_individual, params)            

        # SLC -----------------------------------------------------------------
        params = {}
        for key, values in config_algorithm.items():
            if key == '-normalize':
                continue
            elif key == 'if':
                function = config_algorithm[key]
                return_function = function(params)
                
                if isinstance(return_function, dict):
                    for key_dict, values_dict in return_function.items():
                        # get the individual's index
                        index_individual = dict_indexes.get(algorithm+key_dict)
                        index_cmd, individual, params = get_one_cod(key_dict, values_dict, weka_cmd, index_cmd, individual, index_individual, params) 
            
            elif key == '-K' and isinstance(values, dict): # kernel
                # seta smo-K com 1
                index_individual = dict_indexes.get(algorithm+key)
                individual[index_individual] = 0 # 1
                
                index_cmd = index_cmd + 1 # -K
                # get all kernels
                all_kernels = np.array(list(values.keys())) 
                # get kernel hyperparameter configuration - string
                hips_kernel = weka_cmd[index_cmd]
                hips_kernel = shlex.split(hips_kernel)
                # get the kernel
                index_hips_kernel = 0
                kernel = hips_kernel[index_hips_kernel]
                index_hips_kernel = index_hips_kernel + 1
                
                # sets as 1 the index of the individual referring to the kernel
                index_individual = dict_indexes.get(kernel) 
                individual[index_individual] = 0 # 1
                
                # get kernel configuration               
                config_kernel = values.get(kernel)
                
                for key_kernel, values_kernel in config_kernel.items():
                    # get the individual's index
                    index_individual = dict_indexes.get(kernel+key_kernel)
                    index_hips_kernel, individual, params = get_one_cod(key_kernel, values_kernel, hips_kernel, index_hips_kernel, individual, index_individual, params) 
                    
            else:
                # sets the individual's index to the hyperparameter value
                index_individual = dict_indexes.get(algorithm+key)
                index_cmd, individual, params = get_one_cod(key, values, weka_cmd, index_cmd, individual, index_individual, params)            
                
                    
    return individual


def get_one_cod(key, list_values, command, index_cmd, individual, index_individual, params):
    if list_values.dtype == bool: # list of bool
    
        #if key in command, then key é True
        if index_cmd < len(command) and key == command[index_cmd]:
            individual[index_individual] = 1
            params[key] = True
            index_cmd = index_cmd + 1
        else:
            individual[index_individual] = 0
            params[key] = False
           
    else:
        
        key_hip = command[index_cmd]
        index_cmd = index_cmd + 1
        val_hip = command[index_cmd]
        index_cmd = index_cmd + 1
        
        #print(key_hip, val_hip)
        
        if isNumber(val_hip):
            val_hip = float(val_hip)
            #
            params[key_hip] = val_hip
            index_hip = np.where(list_values == val_hip)[0][0]
            individual[index_individual] = index_hip
        elif len(val_hip.split(' ')) > 1: # words with space
            val_hip = '\''+val_hip+'\''
            #
            i = 0
            for value in val_hip:
                if val_hip in val_hip:
                    break
                i += 1
            index_hip = i
            individual[index_individual] = index_hip
            
        #params[key_hip] = val_hip
        
        #index_hip = np.where(list_values == val_hip)[0][0]
        
        #individual[index_individual] = index_hip
    
    return index_cmd, individual, params