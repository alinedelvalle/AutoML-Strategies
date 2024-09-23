import numpy as np

class IndividualUtils:
    
    #Decoding the individual:
    # Boolean hierparameters may or may not appear in the command, but they occupy positions in the individual
    # Conditional-dependent hierparameters are counted in the individual depending on the condition
    
    # Decoding function for: MLC ensemble, MLC and SLC ensemble
    def command_aux(individual, config_algorithm, i):
        command = ''
        params = {}
        algorithm = ''
        for variable in config_algorithm.keys():
            if variable == 'if': # function
                function = config_algorithm[variable]
                return_function = function(params)
        
                if isinstance(return_function, dict):
                    dictionary = return_function
                    for var in dictionary.keys():
                        all_values = dictionary[var]
                        value = all_values[individual[i]%len(all_values)]
                        params[var] = value
                        
                        if isinstance(value, np.bool_):
                            if value==True:
                                command = command + ' ' + var
                        else:
                                command = command + ' ' + var + ' ' + str(value)
                        
                        i = i + 1  
            else:
                all_values = config_algorithm[variable]
                
                if variable != '-W':
                    value = all_values[individual[i]%len(all_values)]
                    params[variable] = value
                    
                    if isinstance(value, np.bool_):
                        if value==True:
                            command = command + ' ' + variable
                    else:
                        command = command + ' ' + variable + ' ' + str(value)
                else: # -W (algorithm)
                    all_values = list(all_values.keys())
                    value = all_values[individual[i]%len(all_values)]
                    algorithm = value
                    
                i = i + 1
            
        return i, command, algorithm
    
    # -------------------------------------------------------------------------
    
    # Decoding function for SLC
    def command_slc_aux(individual, config, config_algorithm, i):
        command = ''
        params = {}
        for variable in config_algorithm.keys():
            if variable == 'if': # function
                function = config_algorithm[variable]
                return_function = function(params)
        
                if isinstance(return_function, dict):
                    dictionary = return_function
                    for var in dictionary.keys():
                        all_values = dictionary[var]
                        value = all_values[individual[i]%len(all_values)]
                        params[var] = value
                        
                        if isinstance(value, np.bool_):
                            if value==True:
                                command = command + ' ' + var
                        else:
                            command = command + ' ' + var + ' ' + str(value)
                                
                        i = i + 1    
            else: # list ou kernel
                all_values = config_algorithm[variable]
            
                if isinstance(all_values, np.ndarray): # list
                    value = all_values[individual[i]%len(all_values)]
                    params[variable] = value
                    
                    if isinstance(value, np.bool_):
                        if value==True:
                            command = command + ' ' + variable
                    else:
                        command = command + ' ' + variable + ' ' + str(value)
                        
                    i = i + 1
                else: # kernel
                    kernels = list(all_values.keys()) 
                    kernel = kernels[individual[i]%len(kernels)]
                    config_kernel = config.get_sl_kernel_config().get(kernel)                       

                    command = command + ' ' + variable + ' \"' + kernel
                    i = i + 1
                
                    for var in config_kernel.keys():
                        all_values = config_kernel[var]
                        value = all_values[individual[i]%len(all_values)]
                        params[var] = value
                        
                        if isinstance(value, np.bool_):
                            if value==True:
                                command = command + ' ' + var
                        else:
                            command = command + ' ' + var + ' ' + str(value)
                            
                        i = i + 1
                              
                    command = command + '\"'
                    
        return i, command

    # -------------------------------------------------------------------------
    
    # Decodes the individual into weka and meka commands
    def get_commands(config, individual):
        is_pt = False # PT - problem transformation
        is_mlc_ensemble = False
        
        # index 0
        is_normalize = False if individual[0]%2 == 0 else True
        
        # index 1
        index_algorithm = individual[1] 
        algorithm = config.get_ml_algorithms()[index_algorithm%(len(config.get_ml_algorithms()))]
        
        # other indexes
        meka_command = algorithm
        weka_command = None
        i = 2
        
        # ensemble
        if 'meta' in algorithm:
            is_mlc_ensemble = True
            config_ml = config.get_ml_ensemble_config()
            config_algorithm = config_ml.get(algorithm)
            
            i, command, algorithm = IndividualUtils.command_aux(individual, config_algorithm, i)
            meka_command = meka_command + command + ' -W ' + algorithm + ' --'
            config_algorithm = config_algorithm['-W'][algorithm]
            
        # MLC           
        if is_mlc_ensemble == False:
            config_ml = config.get_ml_config() 
            config_algorithm = config_ml.get(algorithm)
        
        if '-W' in config_algorithm.keys():
            is_pt = True
        
        i, command, algorithm = IndividualUtils.command_aux(individual, config_algorithm, i)
        meka_command = meka_command + command
        
        # SLC ensemble
        if is_pt == True:
            weka_command = algorithm + ' --'
            config_algorithm = config_algorithm['-W'][algorithm]
            
            # ensemble
            if 'meta' in algorithm or 'LWL' in algorithm:
                i, command, algorithm = IndividualUtils.command_aux(individual, config_algorithm, i)
                weka_command = weka_command + command + ' -W ' + algorithm + ' --'
                config_algorithm = config_algorithm['-W'][algorithm]
        
            # SLC
            i, command = IndividualUtils.command_slc_aux(individual, config, config_algorithm, i)
            weka_command = weka_command + command
            
        return is_normalize, meka_command, weka_command
                            
    # -------------------------------------------------------------------------
    
    # Get the number of genes that represent the individual
    # Use the auxiliary functions: get_lenght_aux and get_lenght_slc_aux
    def get_lenght_individual(config, individual):
        is_pt = False # PT - problem transformation
        is_mlc_ensemble = False
        
        # index 0 - normalize
        
        # index 1
        index_algorithm = individual[1] 
        algorithm = config.get_ml_algorithms()[index_algorithm%(len(config.get_ml_algorithms()))]
        
        # other indexes
        i = 2
        
        # ensemble
        if 'meta' in algorithm:
            is_mlc_ensemble = True
            config_ml = config.get_ml_ensemble_config()
            config_algorithm = config_ml.get(algorithm)
            
            i, algorithm = IndividualUtils.get_lenght_aux(individual, config_algorithm, i)
            config_algorithm = config_algorithm['-W'][algorithm]
            
        # MLC           
        if is_mlc_ensemble == False:
            config_ml = config.get_ml_config()
            config_algorithm = config_ml.get(algorithm)
        
        if '-W' in config_algorithm.keys():
            is_pt = True
        
        i, algorithm = IndividualUtils.get_lenght_aux(individual, config_algorithm, i)
        
        # SLC ensemble
        if is_pt == True:
            config_algorithm = config_algorithm['-W'][algorithm]
            
            # ensemble
            if 'meta' in algorithm or 'LWL' in algorithm:
                i, algorithm = IndividualUtils.get_lenght_aux(individual, config_algorithm, i)
                config_algorithm = config_algorithm['-W'][algorithm]
        
            # SLC
            i = IndividualUtils.get_lenght_slc_aux(individual, config, config_algorithm, i)
            
        return i # tamanho
    
    # -------------------------------------------------------------------------
    
    # Get the number of genes occupied by: MLC ensemble, MLC and SLC ensemble
    def get_lenght_aux(individual, config_algorithm, i):
        params = {}
        algorithm = ''
        for variable, all_values in config_algorithm.items():
            if variable == 'if': # função
                function = config_algorithm[variable]
                return_function = function(params)
        
                if isinstance(return_function, dict):
                    dictionary = return_function
                    for var, values in dictionary.items():
                        value = values[individual[i]%len(values)]
                        params[var] = value
                        i = i + 1  
            else:
                if variable == '-W': # -W (algorithm)
                    all_values = list(all_values.keys())
                    value = all_values[individual[i]%len(all_values)]
                    algorithm = value
                else: 
                    value = all_values[individual[i]%len(all_values)]
                    params[variable] = value
                    
                i = i + 1
            
        return i, algorithm
    
    # -------------------------------------------------------------------------
    
    # Get the number of genes occupied by SLC algorithms
    def get_lenght_slc_aux(individual, config, config_algorithm, i):
        params = {}
        for variable, all_values in config_algorithm.items():
            if variable == 'if': # funcion
                function = config_algorithm[variable]
                return_function = function(params)
        
                if isinstance(return_function, dict):
                    dictionary = return_function
                    for var, values in dictionary.keys():
                        value = values[individual[i]%len(values)]
                        params[var] = value
                        i = i + 1    
            else: # list ou kernel
            
                if isinstance(all_values, np.ndarray): # list
                    value = all_values[individual[i]%len(all_values)]
                    params[variable] = value  
                    i = i + 1
                    
                else: # kernel
                    kernels = list(all_values.keys()) 
                    kernel = kernels[individual[i]%len(kernels)]
                    config_kernel = config.get_sl_kernel_config().get(kernel)                       
                    i = i + 1
                
                    for var, values in config_kernel.items():
                        value = values[individual[i]%len(values)]
                        params[var] = value
                        i = i + 1
                              
        return i