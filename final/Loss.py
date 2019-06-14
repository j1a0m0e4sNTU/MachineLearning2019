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
        self.weight = torch.Tensor([300, 1, 200])

    def forward(self, pred, gt):
        diff = torch.abs(pred - gt)
        loss = torch.sum(diff @ self.weight) / (gt.size(0) * torch.sum(self.weight))
        return loss

def test():
    loss_func = WMAE()
    pred = torch.randn(8, 3)
    gt = torch.randn(8, 3)
    loss = loss_func(pred, gt)
    print(loss.item())

if __name__ == '__main__':
    test()