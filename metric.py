import torch
from utils import device

def wloss(q_temp, q_target, quantiles, p=2,
          qweight=False, normalize_weight=False, log_weight=False, device=device()):
    """ 
    quantiles : quantile points
    """ 
    q_target = q_target.to(device)
    q_temp = q_temp.to(device)
    quantiles = quantiles.to(device)
    if qweight:
        w = 1/(1-quantiles)
        if log_weight:
            w = torch.log(w)
            
        if normalize_weight:
            w = w/w.sum()
            if p==1:
                l = torch.sum(torch.abs(q_target-q_temp) * w, dim=0)
            else:
                l = torch.sum((q_target-q_temp)**2 * w, dim=0)
        else:
            if p==1:
                l = torch.mean(torch.abs(q_target-q_temp) * w, dim=0)
            else:
                l = torch.mean((q_target-q_temp)**2 * w, dim=0)
    else:
        if p==1:
            l = torch.mean(torch.abs(q_target-q_temp), dim=0)
        else: # p=2
            l = torch.mean((q_target-q_temp)**2, dim=0)
    return l