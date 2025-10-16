from plot_utils import *
from utils import device, Result
import torch
from torch import nn


def test_model_mse(model, TestLoader, device=None):
    """
    run model on TestLoader and report the MSE loss
    
    return:
        - full_output: squeezed numpy array with dimension (N,...)
        - test_loss_mse: mean squared error on TestLoader
    
    """
    if not device:
        device = next(model.parameters()).device
    model.to(device)
    model.eval()
    _, sample_target = next(iter(TestLoader))
    full_output = torch.zeros_like(sample_target[0]).cpu()
    mse_loss = nn.MSELoss()
    running_test_mse_loss = 0.0
    
    with torch.no_grad():
        for _, (test_in, test_tar) in enumerate(TestLoader):
            test_in, test_tar = test_in.to(device), test_tar.to(device)  
            test_output = model(test_in).detach()
            loss = mse_loss(test_output, test_tar)
            running_test_mse_loss += loss.item()
            full_output = torch.cat((full_output,test_output.squeeze().cpu()),dim=0)
    test_mse_loss = running_test_mse_loss/len(TestLoader)
    full_output = full_output[1:].numpy()
    
    return full_output, test_mse_loss

def eval_result(res, X_GRID, X_Samples, N=100, device=device('all'), best_model=False):
    eval_res = Result()
    if best_model:
        model = res.best_model
    else:
        model = res.model
    # evaluate result on a grid
    test_out_grid = test_nn(model, X_GRID, device=device)
    test_out_grid = test_out_grid.reshape((N,N))
    
    # evaluate result on input samples
    test_out = test_nn(model, X_Samples,device=device)

    eval_res.y_grid = test_out_grid
    eval_res.y_samples = test_out
    return eval_res

def eval_result_2d(res, X_GRID, X_Samples, y, N=100, device=device('all'), best_model=False):
    eval_res = Result()
    if best_model:
        model = res.best_model
    else:
        model = res.model
    
    # evaluate result on input samples
    u_samples = test_nn(model, X_Samples, device=device)
    y_samples = y(u_samples)
    eval_res.y_samples = y_samples

    # evaluate result on grid
    u_grid = test_nn(model, X_GRID, device=device)
    u1_grid = u_grid[:,0].reshape((N,N)).detach().cpu().numpy()
    u2_grid = u_grid[:,1].reshape((N,N)).detach().cpu().numpy()
    eval_res.u1_grid = u1_grid
    eval_res.u2_grid = u2_grid
    return eval_res

def test_nn(model, x_test, device=device('all')):
    """
    Test wrapper for a pytorch model 
    """ 

    model.eval()
    with torch.no_grad():
        out_test = model(x_test.to(device)).squeeze().detach()
    return out_test
