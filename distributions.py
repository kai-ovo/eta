""" 
personal distribution class

author: Kai Chang, 
email: kaichang@mit.edu
GitHub: kai-ovo

"""


import torch
from torch.distributions.multivariate_normal import MultivariateNormal

__all__ = ["get_sampler"]


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