import numpy as np

from pymoo.core.mutation import Mutation

from multiobjective.IndividualUtils import IndividualUtils


class MLMutation(Mutation):
    
    def __init__(self, prob, config):
        super().__init__()
        self.prob = prob
        self.config = config


    def _do(self, problem, X, **kwargs):
        
        for individual in X:
            
            n_rand = np.random.rand()  
            
            # Is there mutation?
            if (n_rand < self.prob):
                
                # mutation index
                len_individual = IndividualUtils.get_lenght_individual(self.config, individual)
                index = np.random.randint(0, len_individual)
                
                # new gene value
                value = np.random.randint(0, self.config.get_seed())
                individual[index] = value
                
        return X