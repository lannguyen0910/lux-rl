import torch
import torch.nn as nn
import numpy as np


def compute_loss(pred, target, metrics):
    """
    MSE loss 
    """
    criterion = nn.MSELoss()

    loss = criterion(pred, target)

    pred = torch.argmax(pred, dim=1)

    acc = np.sum(pred.data.cpu().numpy() == target.data.cpu().numpy()
                 ) / len(target.data.cpu().numpy())

    metrics['loss'] += loss.data.cpu().numpy()
    #metrics['acc'] = acc
    return loss
