import numpy as np
import torch
import random
import torch.nn.functional as F


def set_random_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def disable_grads(model):
    for p in model.parameters:
        p.requires_grad = False
    model.eval()


def enable_grads(model):
    for p in model.parameters:
        p.requires_grad = True
    model.train()


def evaluate(model, dloader, device, criterion):
    num_correct = 0
    num_total = 0
    total_loss = 0
    # TODO: Add accuracy from pose estimation here
    with torch.no_grad():
        for i, (x, y, w, meta) in enumerate(dloader):
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)
            model.eval()
            yhat = model(x)
            total_loss += criterion(yhat,y,w)
            num_correct += (yhat.argmax(dim=1) == y).sum().item()
            num_total += x.size(0)
        loss = total_loss / len(dloader)
    acc = 0
    return acc, loss