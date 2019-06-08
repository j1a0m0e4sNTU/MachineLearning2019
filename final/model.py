import torch
import torch.nn as nn

mlp_config ={
    'A': [50, 3],
    'B': [50, 'D', 3]
}

def get_mlp(input_size, config_name):
    config = get_mlp(config_name)
    model = MLP(input_size, config)
    return model

class MLP(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()
        layers = []
        for symbol in config:
            if type(symbol) == int:
                layers += [nn.Linear(input_size, symbol), 
                            nn.BatchNorm1d(symbol),
                            nn.ReLU(inplace= True)]
                input_size = symbol
            if symbol == 'D':
                layers += [nn.Dropout()]

        self.net = nn.Sequential(*layers)
    
    def forward(self, inputs):
        out = self.net(inputs)
        return out

def test_mlp():
    inputs = torch.zeros(8, 200)
    model = MLP(200, mlp_config['A'])
    out = model(inputs)
    print(model)
    print('Output shape: {}'.format(out.size()))

if __name__ == '__main__':
    test_mlp()