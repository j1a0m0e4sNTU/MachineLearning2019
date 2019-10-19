import torch
import torch.nn as nn

mlp_config = {
    'Base': [3],
    'A': [50, 3],
    'B': [600, 3], 
    'C': [600, 200, 3],
    'C_d': [600, 'D', 200, 'D', 3],
    'D': [1024, 512, 256, 3], 
    'E': [1024, 512, 256, 256, 3],
    'F': [1024, 512, 512, 3]
}

rnn_config = {
    'A': {'LSTM': {'hidden': 256, 'layers': 1},'FC':[256, 3], 'bid':False},
    'B': {'LSTM': {'hidden': 180, 'layers': 1},'FC':[360, 3], 'bid':True},
    'C': {'LSTM': {'hidden': 256, 'layers': 1},'FC':[512, 3], 'bid':True},
    'D': {'LSTM': {'hidden': 512, 'layers': 1},'FC':[512, 3], 'bid':False},
    'E': {'LSTM': {'hidden': 256, 'layers': 1},'FC':[256, 256, 3], 'bid':False},
    'F': {'LSTM': {'hidden': 256, 'layers': 2},'FC':[256, 3], 'bid':False},
    'G': {'LSTM': {'hidden': 1024, 'layers': 1},'FC':[1024, 3], 'bid':False},
    'H': {'LSTM': {'hidden': 512, 'layers': 2},'FC':[512, 3], 'bid':False},
    'I': {'LSTM': {'hidden': 1024, 'layers': 2},'FC':[1024, 3], 'bid':False},
    'J': {'LSTM': {'hidden': 512, 'layers': 3},'FC':[512, 3], 'bid':False},
}

def get_mlp(input_size, config_name):
    config = mlp_config[config_name]
    model = MLP(input_size, config)
    return model

def get_rnn(input_dim, name, batch_size):
    model = RNN(input_dim, rnn_config[name], batch_size)
    return model

def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        
        layers = layers[:-2]
        self.net = nn.Sequential(*layers)
    
    def forward(self, inputs):
        out = self.net(inputs)
        return out

def test_mlp():
    import sys
    symbol = sys.argv[1]
    inputs = torch.zeros(8, 200)
    model = MLP(200, mlp_config[symbol])
    out = model(inputs)
    print('Parameter number: {}'.format(parameter_number(model)))
    print('Input  shape: {}'.format(inputs.size()))
    print('Output shape: {}'.format(out.size()))

class RNN(nn.Module):
    def __init__(self, input_dim, config, batch_size):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bidirection = config['bid']
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = config['LSTM']['hidden'],
            num_layers = config['LSTM']['layers'],
            batch_first= True,
            bidirectional = bidirection
        )

        hid_layer = config['LSTM']['layers'] * 2 if bidirection else config['LSTM']['layers']
        self.state_shape = (hid_layer, batch_size, config['LSTM']['hidden'])

        fc_layers = []
        for i in range(len(config['FC']) - 1):
            fc_layers.append(nn.Linear(config['FC'][i], config['FC'][i + 1]))
            fc_layers.append(nn.ReLU(inplace= True))
        fc_layers = fc_layers[:-1]
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, inputs):
        # inputs size: (batch_size, sequence_len, feature_number)
        # output size of LSTM: (batch_size, sequence_len,, hidden_size)
        states = (torch.zeros(self.state_shape).to(self.device), torch.zeros(self.state_shape).to(self.device))
        r_out, new_states = self.lstm(inputs, states)
        last_out = r_out[:, -1, :].squeeze(1)
        out = self.classifier(last_out)
        return out

def test_rnn():
    import sys
    symbol = sys.argv[1]
    model = get_rnn(10, symbol, 8)
    inputs = torch.zeros(8, 20, 10)
    out = model(inputs)
    print('Parameter number: {}'.format(parameter_number(model)))
    print('Input  shape: {}'.format(inputs.size()))
    print('Output shape: {}'.format(out.size()))

if __name__ == '__main__':
    test_rnn()
