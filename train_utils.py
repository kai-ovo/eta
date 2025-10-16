from tqdm import tqdm
import torch.nn as nn
import torch
from copy import deepcopy
import numpy as np
from scipy.stats import multivariate_normal
from metric import wloss
from utils import *

def train_nn(model, 
             x_train, 
             y_train, 
             loss_fun,
             optim,
             epochs,
             batch_size,
             device,
             save_model=False,
             save_path=None,
             **opt_params):
    """ 
    optim: torch.optim object
    **opt_params: parameters for optim
    """
    
    model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    optimizer = optim(model.parameters(), **opt_params)
    criterion = loss_fun()

    # get batch info
    N = x_train.shape[0]
    num_batch = N // batch_size if (N % batch_size) == 0 else N // batch_size + 1

    # record result
    res = Result()    
    losses = []
    
    if save_model:
        assert save_path is not None
    
    min_loss = float("Inf")
    best_model = None
    model.train()
    for i in tqdm(range(epochs)):

        for b in range(num_batch):
            # fetch data
            x_batch = x_train[b*batch_size:(b+1)*batch_size,]
            y_batch = y_train[b*batch_size:(b+1)*batch_size,]

            # zero out gradient
            optimizer.zero_grad()
            
            # forward pass
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()

            # record loss
            losses.append(loss.detach().item())

            # update
            optimizer.step()
            
            # update model to save
            if losses[-1] < min_loss:
                min_loss = losses[-1]
                best_model = deepcopy(model)
    
    if save_model:
        torch.save(best_model.state_dict(), save_path)
        
    res.model = best_model
    res.loss = losses
    return res

def train_eta_xy(model,
                 x_train,
                 y_train,
                 Xall,
                 Yall,
                 quantiles,
                 optim,
                 epochs,
                 batch_size,
                 device,
                 _lamb=1.0,
                 omega=1,
                 **opt_params):
    """ 
    optim: torch.optim object
    **opt_params: parameters for optim

    Xall : all input data to compute model quantiles
    e2a_loss : a tail-aware loss function
    biasing_dist: reference pdf to move towards
                  should be heavy-tailed to contain extremes

    """
    
    model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    Xall = Xall.to(device)
    Yall = Yall.to(device)
    quantiles = quantiles.to(device)
    optimizer = optim(model.parameters(), **opt_params)
    mse_loss = nn.MSELoss()

    # define eta loss
    Ytail = torch.quantile(Yall, quantiles, dim=0)
    eta_loss = lambda y: wloss(y, Ytail, quantiles, p=1)

    # function to update tail index
    def update_idx(Y):
        _, indices = torch.sort(Y)
        quantile_positions = (quantiles * (Y.size(0) - 1)).floor().long()
        return indices[quantile_positions]

    # initialize tail index
    model.eval()
    with torch.no_grad():
        Yall_temp = model(Xall).squeeze()
        tail_idx = update_idx(Yall_temp)
    
    # get batch info
    N = x_train.shape[0]
    num_batch = N // batch_size if (N % batch_size) == 0 else N // batch_size + 1

    # record results
    res = Result()
    eta_losses = []
    mse_losses = []
    best_model = None
    best_eta_loss = float("Inf")
    
    for i in tqdm(range(epochs)):
        model.train()
        for b in range(num_batch):
            # fetch data
            x_batch = x_train[b*batch_size:(b+1)*batch_size,]
            y_batch = y_train[b*batch_size:(b+1)*batch_size,]

            # zero out gradient
            optimizer.zero_grad()
            
            # forward pass
            batch_out = model(x_batch).squeeze()
            mse_loss_val = mse_loss(batch_out, y_batch)

            # tail forward pass
            Xtail = Xall[tail_idx]
            Ytail_temp = model(Xtail).squeeze()
            eta_loss_val = eta_loss(Ytail_temp)

            # total loss
            loss = mse_loss_val + _lamb*eta_loss_val

            # update
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_epoch = model(x_train).squeeze()
            mse_loss_epoch = mse_loss(y_epoch, y_train).detach().item()
            mse_losses.append(mse_loss_epoch)

            Yall_epoch = model(Xall).squeeze()
            Ytail_epoch = torch.quantile(Yall_epoch, quantiles, dim=0)
            eta_loss_epoch = eta_loss(Ytail_epoch).detach().item()
            eta_losses.append(eta_loss_epoch)

            if eta_loss_epoch < best_eta_loss:
                best_eta_loss = eta_loss_epoch
                best_model = deepcopy(model)

            if (i+1) % omega == 0:
                tail_idx = update_idx(Yall_epoch)

    res.model = model
    res.best_model = best_model
    res.eta_loss = eta_losses
    res.mse = mse_losses
    return res

def train_eta_xuy(model,
                  x_train,
                  u_train,
                  Xall,
                  Yall,
                  y, 
                  quantiles,
                  optim,
                  epochs,
                  batch_size,
                  device,
                  _lamb=1.0,
                  omega=1,
                  **opt_params):
    """ 
    optim: torch.optim object
    **opt_params: parameters for optim

    Xall : all input data to compute model quantiles
    e2a_loss : a tail-aware loss function
    biasing_dist: reference pdf to move towards
                  should be heavy-tailed to contain extremes

    """
    
    model.to(device)
    x_train = x_train.to(device)
    u_train = u_train.to(device)
    Xall = Xall.to(device)
    Yall = Yall.to(device)
    quantiles = quantiles.to(device)
    optimizer = optim(model.parameters(), **opt_params)
    mse_loss = nn.MSELoss()

    # define eta loss
    Ytail = torch.quantile(Yall, quantiles, dim=0)
    eta_loss = lambda y: wloss(y, Ytail, quantiles, p=1)

    # function to update tail index
    def update_idx(Y):
        _, indices = torch.sort(Y)
        quantile_positions = (quantiles * (Y.size(0) - 1)).floor().long()
        return indices[quantile_positions]

    # initialize tail index
    model.eval()
    with torch.no_grad():
        Uall_temp = model(Xall).squeeze()
        Yall_temp = y(Uall_temp)
        tail_idx = update_idx(Yall_temp)
    
    # get batch info
    N = x_train.shape[0]
    num_batch = N // batch_size if (N % batch_size) == 0 else N // batch_size + 1

    # record results
    res = Result()
    eta_losses = []
    mse_losses = []
    best_model = None
    best_eta_loss = float("Inf")
    
    for i in tqdm(range(epochs)):
        model.train()
        for b in range(num_batch):
            # fetch data
            x_batch = x_train[b*batch_size:(b+1)*batch_size,]
            u_batch = u_train[b*batch_size:(b+1)*batch_size,]

            # zero out gradient
            optimizer.zero_grad()
            
            # forward pass
            batch_out = model(x_batch).squeeze()
            mse_loss_val = mse_loss(batch_out, u_batch)

            # tail forward pass
            Xtail = Xall[tail_idx]
            Utail_temp = model(Xtail).squeeze()
            Ytail_temp = y(Utail_temp)
            eta_loss_val = eta_loss(Ytail_temp)

            # total loss
            loss = mse_loss_val + _lamb*eta_loss_val

            # update
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            u_epoch = model(x_train).squeeze()
            mse_loss_epoch = mse_loss(u_epoch, u_train).detach().item()
            mse_losses.append(mse_loss_epoch)

            Uall_epoch = model(Xall).squeeze()
            Yall_epoch = y(Uall_epoch)
            Ytail_epoch = torch.quantile(Yall_epoch, quantiles, dim=0)
            eta_loss_epoch = eta_loss(Ytail_epoch).detach().item()
            eta_losses.append(eta_loss_epoch)

            if eta_loss_epoch < best_eta_loss:
                best_eta_loss = eta_loss_epoch
                best_model = deepcopy(model)

            if (i+1) % omega == 0:
                tail_idx = update_idx(Yall_epoch)

    res.model = model
    res.best_model = best_model
    res.eta_loss = eta_losses
    res.mse = mse_losses
    return res

def grf_pretrain(grf, model, epochs=1000, n_grid=100, grid_step=1, nbatch=1, device=device()):
    Y_pre = grf[::grid_step, ::grid_step].float()
    Y_pre = Y_pre.reshape(Y_pre.shape[0]*Y_pre.shape[1], -1).squeeze()
    X1, X2 = torch.meshgrid(torch.linspace(-6,6, n_grid), \
                              torch.linspace(-6,6, n_grid), indexing='ij')
    X_pre = torch.cat((X1.reshape(-1,1),X2.reshape(-1,1)), dim=1)
    loss_fun = nn.MSELoss
    optim = torch.optim.Adam
    batch_size = n_grid**2//nbatch 
    res = train_nn(model, X_pre, Y_pre, loss_fun, optim, epochs, batch_size, device)
    res.grf = grf
    res.X1 = X1
    res.X2 = X2
    return res 

# a class to record result
class Result(object):

    def __init__(self) -> None:
        pass


def generate_2d_gaussian(seed=42, grid_size=100, domain=(-6, 6), peak_value=0.8, sigma=1.0):
    """
    Generate a 2D Gaussian function with a peak value around 0.8 centered at a random point.
    
    Parameters:
    -----------
    seed : int
    grid_size : int
    domain : tuple
    peak_value : float
    sigma : float
        
    Returns:
    --------
    numpy.ndarray
        2D array of shape (grid_size, grid_size) containing the function values
    tuple
        The center coordinates (x1_center, x2_center)
    """
    # Set random seed
    np.random.seed(seed)
    
    # Generate random center point
    x1_center = np.random.uniform(domain[0], domain[1])
    x2_center = np.random.uniform(domain[0], domain[1])
    
    # Create grid
    x1 = np.linspace(domain[0], domain[1], grid_size)
    x2 = np.linspace(domain[0], domain[1], grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Stack coordinates
    pos = np.dstack((X1, X2))
    
    # Create multivariate normal distribution
    rv = multivariate_normal([x1_center, x2_center], sigma * np.eye(2))
    
    # Evaluate pdf on grid
    pdf = rv.pdf(pos)
    
    # Scale to desired peak value
    max_pdf = pdf.max()
    scaled_pdf = pdf * (peak_value / max_pdf)
    
    return torch.from_numpy(scaled_pdf), (x1_center, x2_center)
