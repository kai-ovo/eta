""" 
ML utility functions

author: Kai Chang, 
email: kaichang@mit.edu
GitHub: kai-ovo

"""

import numpy as np
import random, os
import torch 
from timeit import default_timer as timer
from torch.nn import functional as F 
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


def global_seed(seed: int):
    """
    Set seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_path(path):
    """
    if the directory does not exist, then create it
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")

def time_func(func, *args, **kwargs):
    """Calls func with given args and returns a (seconds, res) tuple.
    :param func: function to be evaluated
    :param args: args for func
    :param kwds: kwds for func
    :return: a tuple: (seconds, func(*args, **kwds)
    """
    tic = timer()
    res = func(*args, **kwargs)
    toc = timer()
    return toc - tic, res

def device(gpu='all'):
    if gpu=='all':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: 
        return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

class HyperParams(object):

    """
    A class for saving hyperparameters

    Example:
    >>> params = {'k1' : val1, 
                  'k2' : val2}
    >>> hp = HyperParams(**params)
    >>> hp.k1
    val1
    >>> hp.k2
    val2

    """

    def __init__(self, **params):
        self.__dict__.update(params) 

    def update(self, **params):
        """
        update attributes (in this case, hyperparameters)
        """
        for key, value in params.items():
            setattr(self, key, value)

def choose_act(activation):
    if activation is None:
        return lambda x: x
    elif activation == 'relu':
        return lambda x: F.relu(x, inplace=False)
    elif activation == 'gelu':
        return F.gelu
    elif activation == 'tanh':
        return F.tanh
    elif activation == 'elu':
        return F.elu
    elif activation == 'leakyrelu':
        return lambda x: F.leaky_relu(x, inplace=False)
    elif activation == 'sigmoid':
        return F.sigmoid
    elif activation == 'silu':
        return F.silu
    elif activation=='psilu':
        return lambda x : psilu(x, a=4)
    elif activation == 'softplus':
        return lambda x : F.softplus(x, beta=8)
    else:
        raise ValueError(f"{activation} is not implemented!")

def psilu(x, a=4):
    return x * F.sigmoid(a*x)
        
def init_linear(m, init):
    """ 
    initialize linear layer weights
    """
    initialize = choose_init(init)
    if isinstance(m, nn.Linear):
        initialize(m.weight)
        m.bias.data.fill_(0.01)


def choose_init(init:str):
    if init=='xavier uniform':
        return torch.nn.init.xavier_uniform_
    elif init == 'xavier normal':
        return torch.nn.init.xavier_normal_
    elif init == 'kaiming uniform':
        return torch.nn.init.kaiming_uniform_
    elif init == 'kaiming normal':
        return torch.nn.init.kaiming_normal_
    else:
        raise ValueError(f"{init} is not an available initialization")

# a class to record result
class Result(object):

    def __init__(self) -> None:
        pass



class mvn(MultivariateNormal):
    """
    rewrite the torch mvn distribution to incorporate pdf
    """

    def __init__(self, loc, cov, validate_args=None):
        super().__init__(loc, covariance_matrix=cov, validate_args=validate_args)
    
    def pdf(self, value):
        log_pdf = self.log_prob(value)
        return torch.exp(log_pdf)


def get_sampler(mu, cov, option):
    """ 
    input: 
        option: sampler options 
    
    output: 
        torch.distributions.distribution.Distribution Object
    
        available functions:  
            output.cdf(value : tensor)
            output.sample(sample_shape : shape) -> tensor with corresponding shape
            output.sample_n(n : sample_size) -> (n,)
            output.log_prob(value : tensor) -> 

    """
    if option=="mvn":
        return mvn(mu, cov)
    
def input(mu, cov, option):
    """ 
    input:
        option: input distribution type

    output:
        sampler object 

    """
    return get_sampler(mu, cov, option)