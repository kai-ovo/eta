import torch
import numpy as np
from KDEpy import FFTKDE
from scipy.stats import gaussian_kde

__all__ = ["get_data_pdf",
           "custom_KDE"]

def get_data_pdf(data, data_all, weights=None):
    """ 
    Wrapper for computing output pdf 
    
    outputs:
        y_eval: values of y evaluated by the KDE estimated pdf
                lies in the value range of data_all
    """
    if weights is not None:
        if type(weights) is torch.Tensor:
            weights = weights.numpy()
    
    if type(data_all) is torch.Tensor:
        data_all = data_all.numpy()
    
    data_min =  np.min([data.detach().cpu().numpy().min(),data_all.min()]) - 1e-8
    data_max =  np.max([data.detach().cpu().numpy().max(),data_all.max()]) + 1e-8
    y_eval = np.linspace(data_min, data_max, 1000)
    y_pdf = custom_KDE(data.detach().cpu().numpy(), weights=weights, bw=None)
    py = y_pdf.evaluate(y_eval)
    
    py = py * (y_eval < data_all.max())
    py = py * (y_eval > data_all.min())
    py = np.trim_zeros(py)
    y_eval = y_eval * (y_eval < data_all.max())
    y_eval = y_eval * (y_eval > data_all.min())
    y_eval = np.trim_zeros(y_eval)

    return y_eval, py, y_pdf

def custom_KDE(data, weights=None, bw=None):
    """ 
    Note that to evaluate pdf of data of dimension > 1 with KDE, we need to first run 
    KDE on a regular grid and then use interpolation to evaluate on arbitrary data points

    check https://kdepy.readthedocs.io/en/latest/examples.html#the-effect-of-norms-in-2d
    and   https://kdepy.readthedocs.io/en/latest/examples.html#fast-evaluation-on-a-non-equidistant-grid
    """
    data = data.flatten()
    if bw is None:
        try:
            sc = gaussian_kde(data, weights=weights)
            bw = np.sqrt(sc.covariance).flatten()
            # Ensure that bw is a scalar value
            if np.size(bw) == 1:
                bw = bw[0]
            else:
                raise ValueError("The bw must be a number.")
        except:
            bw = 1
        if bw < 1e-8:
            bw = 1
            
    return FFTKDE(bw=bw).fit(data, weights=weights)