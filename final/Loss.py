import torch
import torch.nn as nn

class MSE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, gt):
        loss = torch.mean(torch.pow(pred - gt, 2))
        return loss

class WMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = [300, 1, 200]

    def forward(self, pred, gt):
        diff = torch.abs(pred - gt)
        loss = 0
        for i in range(3):
            loss += torch.sum(diff[:, i] * self.weight[i])
        loss /= (gt.size(0) * sum(self.weight))
        return loss

class NAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        diff = torch.abs(pred - gt)
        loss = torch.mean(torch.abs(diff / gt))
        return loss 

class ABS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        loss = torch.mean(torch.abs(pred - gt))
        return loss

def test():
    loss_func = ABS()
    pred = torch.randn(8, 3)
    gt = torch.randn(8, 3)
    loss = loss_func(pred, gt)
    print(loss.item())

if __name__ == '__main__':
    test()
