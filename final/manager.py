import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from util import evaluate

class Manager():
    def __init__(self, model, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.load:
            model.load_state_dict(torch.load(args.load))
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr= args.lr)
        self.metric = nn.MSELoss()
        self.epoch_num = args.epoch
        self.batch_size = args.bs
        self.save = args.save
        self.csv = args.csv
        self.best = {'epoch': 0, 'wmae': 9999}
        
        if args.record:
            self.record_file = open(args.record, 'w')
            self.record('Info: {}\n'.format(args.info))
            self.record('Model: \n {} \n'.format(self.model.__str__()))
            self.record('=========================')
    
    def record(self, info):
        print(info)
        self.record_file.write('{}\n'.format(info))

    def train(self, train_data, valid_data):
        for epoch in range(self.epoch_num):
            self.model.train()
            train_wmae, train_nae = 0, 0
            for step, (train_x, train_y) in enumerate(train_data):
                train_x = train_x.to(self.device)
                train_y = train_y.to(self.device)
                out = self.model(train_x)
                self.optimizer.zero_grad()
                loss = self.metric(out, train_y)
                loss.backward()
                self.optimizer.step()
                
                out = out.detach().cpu().numpy()
                train_y = train_y.detach().cpu().numpy()
                wmae, nae = evaluate(out, train_y)
                train_wmae += wmae 
                train_nae += nae
            
            train_wmae /= (step + 1)
            train_nae /= (step + 1)
            valid_wmae, valid_nae = self.validate(valid_data)

            best_info = ''
            if valid_wmae < self.best['wmae']:
                self.best['epoch'] = epoch
                self.best['wmae'] = valid_wmae
                best_info = '* Best WMAE *'

            info = 'Epoch {} | Train WMAE: {} Train NAE: {} Valid WMAE: {} Valid NAE: {}  {}'.format(
                epoch, train_wmae, train_nae, valid_wmae, valid_nae, best_info
            )
            self.record(info)
                
    def validate(self, valid_data):
        self.model.eval()
        valid_wmae, valid_nae = 0, 0
        for step, (valid_x, valid_y) in enumerate(valid_data):
            valid_x = valid_x.to(self.device)
            valid_y = valid_y.to(self.device)
            out = self.model(valid_x)
            
            out = out.detach().cpu().numpy()
            valid_y = valid_y.detach().cpu().numpy()
            wmae, nae = evaluate(out, valid_y)
            valid_wmae += wmae
            valid_nae  += nae
        
        valid_wmae /= (step + 1)
        valid_nae /= (step + 1)
        return valid_wmae, valid_nae

    def predict(self, test_data):
        self.model.eval()
        
