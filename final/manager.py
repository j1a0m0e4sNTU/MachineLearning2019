import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from util import evaluate, write_submission
from Loss import *

class Manager():
    def __init__(self, model, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.load:
            model.load_state_dict(torch.load(args.load))
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr= args.lr)
        self.loss_func = ABS()
        self.epoch_num = args.epoch
        self.batch_size = args.bs
        self.save = args.save
        self.csv = args.csv
        self.best = {'loss': 9999, 'wmae': 9999, 'nae': 9999}
        self.record_file = None
        
        if args.record:
            self.record_file = open(args.record, 'w')
            self.record('Info: {}\n'.format(args.info))
            self.record('Model: \n {} \n'.format(self.model.__str__()))
            self.record('=========================')
    
    def record(self, info):
        print(info)
        if self.record_file:
            self.record_file.write('{}\n'.format(info))

    def train(self, train_data, valid_data):
        for epoch in range(self.epoch_num):
            self.model.train()
            train_loss = 0
            train_wmae, train_nae = 0, 0
            for step, (train_x, train_y) in enumerate(train_data):
                train_x = train_x.to(self.device)
                train_y = train_y.to(self.device)
                out = self.model(train_x)
                self.optimizer.zero_grad()
                loss = self.loss_func(out, train_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                out = out.detach().cpu().numpy()
                train_y = train_y.detach().cpu().numpy()
                wmae, nae = evaluate(out, train_y)
                train_wmae += wmae 
                train_nae += nae
            
            train_loss /= (step + 1)
            train_wmae /= (step + 1)
            train_nae /= (step + 1)
            valid_loss, valid_wmae, valid_nae = self.validate(valid_data)

            best_info = ''
            if valid_loss < self.best['loss']:
                self.best['loss'] = valid_loss
                best_info += ' Loss '
            if valid_wmae < self.best['wmae']:
                self.best['wmae'] = valid_wmae
                best_info += ' WMAE '
            if valid_nae < self.best['nae']:
                self.best['nae'] = valid_nae
                best_info += ' NAE '

            self.record('\n------------  Epoch {} ----------- Best: {}'.format(epoch, best_info))
            self.record('Train => Loss: {:.5f} | WMAE: {:.5f} | NAE: {:.5f}'.format(train_loss, train_wmae, train_nae))
            self.record('Valid => Loss: {:.5f} | WMAE: {:.5f} | NAE: {:.5f}'.format(valid_loss, valid_wmae, valid_nae))
            
            if self.save and 'NAE' in best_info:
                torch.save(self.model.state_dict(), self.save)

        self.record('\n========== Best record ==========')
        self.record('Loss: {:.5f} | WMAE: {:.5f} | NAE: {:.5f}'.format(self.best['loss'], self.best['wmae'], self.best['nae']))

    def validate(self, valid_data):
        self.model.eval()
        valid_loss = 0
        valid_wmae, valid_nae = 0, 0
        for step, (valid_x, valid_y) in enumerate(valid_data):
            valid_x = valid_x.to(self.device)
            valid_y = valid_y.to(self.device)
            out = self.model(valid_x)
            loss = self.loss_func(out, valid_y)
            valid_loss += loss.item()

            out = out.detach().cpu().numpy()
            valid_y = valid_y.detach().cpu().numpy()
            wmae, nae = evaluate(out, valid_y)
            valid_wmae += wmae
            valid_nae  += nae
        
        valid_loss /= (step + 1)
        valid_wmae /= (step + 1)
        valid_nae /= (step + 1)
        return valid_loss, valid_wmae, valid_nae

    def predict(self, test_data):
        self.model.eval()
        pred = None
        for step, test_x in enumerate(test_data):
            test_x = test_x.to(self.device)
            out = self.model(test_x)
            
            if step == 0:
                pred = out
            else:
                pred = torch.cat([pred, out], 0)

        pred = pred.detach().cpu().numpy()
        write_submission(pred, self.csv)
        
