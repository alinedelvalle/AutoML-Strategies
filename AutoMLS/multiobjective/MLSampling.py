import numpy as np

from pymoo.core.sampling import Sampling

class MLSampling(Sampling):
    
    
    def __init__(self, config, n_gene, is_dataset_sparse=False):
        super().__init__()
        self.config = config # hyperparameter configuration
        self.n_gene = n_gene # gene number
        self.is_dataset_sparse = is_dataset_sparse # sparse dataset?
        
     
    # returns the population with n_samples individuals   
    def _do(self, problem, n_samples, **kwargs):
        # normalization
        if self.is_dataset_sparse == False:
            norm = np.random.randint(2, size=n_samples)
        else:
            # we do not normalize sparse dataset
            norm = np.random.randint(1, size=n_samples)
        
        # selects values ​​of the other genes
        algs_hips = np.random.randint(self.config.get_seed(), size=(n_samples, self.n_gene - 1))
        
        # joins selected algorithms and hyperparameters
        X = np.column_stack([norm, algs_hips])

        return X
