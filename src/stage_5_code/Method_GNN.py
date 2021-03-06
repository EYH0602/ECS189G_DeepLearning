"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from src.base_class.method import method
from src.stage_5_code.Evaluate_Accuracy import EvaluateAccuracy
from src.stage_5_code.Training_Conv_Plotter import Plotter
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("training on: " + device.type)


class MethodGNN(method, torch.nn.Module):
    data = None
    mask = None
    max_epoch = 5000
    learning_rate = 1e-5
    plotter: Plotter = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        torch.nn.Module.__init__(self)

        # ! temp settings for cora dataset
        self.conv1 = GCNConv(1433, 789)
        self.conv2 = GCNConv(789, 7)

    def forward(self, x, edge):
        """Forward propagation"""
        
        x = self.conv1(x, edge)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge)
        
        y_pred = F.log_softmax(x, dim=1)
        return y_pred
                
    def train(self, X, edge, y):
        
        # check for plot setting
        if not self.plotter:
            raise RuntimeWarning("Plotter not defined.")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = torch.nn.NLLLoss()
        accuracy_evaluator = EvaluateAccuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            optimizer.zero_grad()

            X_train = torch.FloatTensor(X).to(device)
            edge = torch.LongTensor(edge).to(device)
            
            y_pred = self.forward(X_train, edge)
            y_true = torch.LongTensor(np.array(y))
           
           
            train_loss = loss_function(y_pred[self.mask['idx_train']], y_true[self.mask['idx_train']])
            train_loss.backward()
            optimizer.step()

            accuracy_evaluator.data = {
                'true_y': y_true[self.mask['idx_train']],
                'pred_y': y_pred[self.mask['idx_train']].cpu().max(1)[1]
            }
            
            if epoch % 100 == 0:
                print(
                    'Epoch:', epoch,
                    'Accuracy:', accuracy_evaluator.evaluate_accuracy(),
                    'Precision', accuracy_evaluator.evaluate_precision(),
                    'Recall', accuracy_evaluator.evaluate_recall(),
                    'F1', accuracy_evaluator.evaluate_f1(),
                    'Loss:', train_loss.item()
                )
            
            # add to graph if avaliable
            if self.plotter:
                self.plotter.train_loss.append(train_loss.item())
                self.plotter.train_acc.append(accuracy_evaluator.evaluate_accuracy(verbose=False))
                
                # also plot validation
                val_loss = loss_function(y_pred[self.mask['idx_val']], y_true[self.mask['idx_val']])
                accuracy_evaluator.data = {
                    'true_y': y_true[self.mask['idx_val']],
                    'pred_y': y_pred[self.mask['idx_val']].cpu().max(1)[1]
                }
                self.plotter.val_loss.append(val_loss.item())
                self.plotter.val_acc.append(accuracy_evaluator.evaluate_accuracy(verbose=False))
                

    def test(self, X, edge):
        # do the testing, and result the result
        X_test = torch.FloatTensor(X).to(device)
        edge = torch.LongTensor(edge).to(device)
        y_pred = self.forward(X_test, edge)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.cpu().max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['X'], self.data['edge'], self.data['y'])
        print('--start testing...')
        pred_y = self.test(self.data['X'], self.data['edge'])
        return {'pred_y': pred_y, 'true_y': self.data['y']}
