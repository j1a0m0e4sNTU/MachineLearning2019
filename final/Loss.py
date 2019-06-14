import torch
import torch.nn as nn

class MSE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, gt):
        loss = torch.mean(torch.pow(pred - gt, 2))
        return loss

def test():
    loss_func = MSE()
    pred = torch.randn(8, 10, 10)
    gt = torch.randn(8, 10, 10)
    loss = loss_func(pred, gt)
    print(loss.item())


if __name__ == '__main__':
    test()