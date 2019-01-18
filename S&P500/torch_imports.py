import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np

class ModelData():
    def __init__(self, trn, val, y, drop, ds, op):
        trn_y = trn[y].copy()
        trn_X = trn.drop([y] + drop, axis=1)
        val_y = val[y]
        val_X = val.drop([y] + drop, axis=1)

        op = op.fit(trn_X)
        trn_X = op.transform(trn_X)
        val_X = op.transform(val_X)

        self._trnset = ds(trn_X, trn_y)
        self._valset = ds(val_X, val_y)
        self.data = None

    def makeDataLoaders(self, bs=32, shuffle=True, num_workers=0):
        trn_dl = DataLoader(self._trnset, batch_size=bs,
                            shuffle=shuffle, num_workers=num_workers)
        val_dl = DataLoader(self._valset, batch_size=bs,
                            shuffle=shuffle, num_workers=num_workers)
        self.data = (trn_dl, val_dl)


def showFirstBatch(dl: DataLoader):
    it = iter(dl)
    x = next(it)
    print(x)
    return x


class Stepper():
    def __init__(self, model, opt, crit):
        self.model, self.opt, self.crit = model, opt, crit
        self.reset(True)

    def reset(self, train=True):
        if train:
            if isinstance(self.model, nn.Module):
                self.model.train()
            for l in self.model.modules():
                l.train()
        else:
            # val
            self.m.eval()

    def step(self, xs, y, epoch):
        def torch_item(x): return x.item() if hasattr(x, 'item') else x[0]
        xtra = []
        self.opt.zero_grad() # best practice to run zero_grad on optimizer
        output = self.model(xs)
        loss = raw_loss = self.crit(output, y)
        loss.backward()
        if 'wd' in self.opt.param_groups[0] and self.opt.param_groups[0]['wd'] != 0:
            # Weight decay out of the loss. After the gradient computation but before the step.
            for group in self.opt.param_groups:
                lr, wd = group['lr'], group['wd']
                for p in group['params']:
                    if p.grad is not None:
                        p.data = p.data.add(-wd * lr, p.data)
        self.opt.step()
        return torch_item(raw_loss.data)

    def evaluate(self, xs, y):
        preds = self.model(xs)
        if isinstance(preds, tuple):
            preds = preds[0]
        return preds, self.crit(preds, y)

def to_np(v):
    '''returns an np.array object given an input of np.array, list, tuple, torch variable or tensor.'''
    if isinstance(v, float): return np.array(v)
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, (list,tuple)): return [to_np(o) for o in v]
    if isinstance(v, torch.FloatTensor): v=v.float()
    return v.cpu().numpy()