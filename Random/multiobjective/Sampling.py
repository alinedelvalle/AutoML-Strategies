import numpy as np


class Sampling:
    
    
    def __init__(self, config, is_dataset_sparse=False):
        super().__init__()
        self.n_gene = 28 # gene numbers
        self.config = config # hyperparameter configuration
        self.is_dataset_sparse = is_dataset_sparse # sparse dataset?
    
        
    # returns 1 encoded MLC algorithm 
    def do(self):
        # select normalization
        if self.is_dataset_sparse == False:
            norm = np.random.randint(2)
        else:
            norm = np.random.randint(1) 
        
        # selects values ​​of the other genes
        algs_hips = np.random.randint(self.config.get_seed(), size=(self.n_gene - 1))
        
        # joins selected algorithms and hyperparameters
        X = np.append(norm, algs_hips)
        
        return X