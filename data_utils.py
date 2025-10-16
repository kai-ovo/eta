from distributions import *
import torch
from scipy.stats.qmc import LatinHypercube as LH
import numpy as np
from scipy.stats.qmc import scale 
import h5py
from torch.distributions.normal import Normal

###### toy dataset construction util functions

def input(mu, cov, option):
    """ 
    input:
        option: input distribution type

    output:
        sampler object 

    """
    return get_sampler(mu, cov, option)

def true_map(locs, covs, heights, option='mvn'):
    """ 
    analytical mapping from input space to output QoI
    ideally some functions with several spikes in the tail region

    input:
        locs: center of the spikes, (m,d)
            m = number of spikes
            d = dimension of the input space
        vars: parameter that controls the shape of the spikes, (m,d,d)
        heights: possible scaling factor to scale the spike, (m,)
        option: type of spike function
            default='mvn' : multivariate normal
    
    output:
        mapping: true input-output map
            input: values, torch.Tensor, (n,d)
                   n: number of samples, d: dimension
    """
    if not isinstance(locs, torch.Tensor):
        locs = torch.Tensor(locs)
    m,d = locs.size()
    if not isinstance(covs, torch.Tensor):
        covs = torch.Tensor(covs)

    samplers = []
    for i in range(m):
        loc = locs[i,:]
        cov = covs[i,:,:]
        samplers.append(get_sampler(loc, cov, option))
    
    def mapping(values):
        n = values.size(dim=0)
        res = torch.zeros(n)
        for i,s in enumerate(samplers):
            h = heights[i]
            res += h * s.pdf(values)
        
        return res
    
    return mapping

def get_lhs(dim, N, seed=0, l_bounds=None, u_bounds=None, to_tensor=True):
    """
    get Latin Hypercube samples

    input:
        dim: data dimensions
        N: #samples
        l_bounds: lower bounds for LH samples
        u_bounds: upper bounds for LH samples
        to_tensor: flag for returning samples as torch.tensor
    """
    if l_bounds is None:
        assert u_bounds is None 
        l_bounds = np.zeros(dim)
        u_bounds = np.ones(dim)

    sampler = LH(d = dim, seed=seed)
    samples = sampler.random(n=N)
    samples = scale(samples, l_bounds, u_bounds)

    if to_tensor:
        return torch.from_numpy(samples)
    else:
        return samples

def lhs_to_gaussian(N=None, d=None, samples=None, scale=1.0, seed=0, to_tensor=True):
    """
    pushes LHS samples on [0,1]^d to Standard Gaussian in R^d 
    
    """
    if samples == None:
        assert d != None
        assert N != None
        sampler = LH(d = d, seed=seed)
        samples = sampler.random(n=N)
    if type(samples) is np.ndarray:
        samples = torch.from_numpy(samples)
        
    # apply inverse cdf
    samples = Normal(torch.tensor([0.0]), torch.tensor([scale])).icdf(samples) 
    return samples
    
def read_hdf5(filename, to_tensor=True):
    with h5py.File(filename, 'r') as File:
        data = File['data'][:]
    
    if to_tensor:
        return torch.from_numpy(data)
    return data