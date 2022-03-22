import torch.nn as nn
import torch

def L1( model):
    loss = 0
    for param in model.parameters():
        loss += torch.sum(abs(param))
    return loss


def L2(model):
    loss = 0
    for param in model.parameters():
        try:
            loss += torch.sum(param ** 2)
        except:
            print()
    return loss


def cross_entropy(p, gt):
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(p, gt)
    return loss