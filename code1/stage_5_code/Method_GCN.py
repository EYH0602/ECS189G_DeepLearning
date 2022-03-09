from code1.base_class.method import method
import torch
from code1.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Method_GCN(method, nn.Module):
    data = None
    max_epoch = 100
    learning_rate = 1e-4
    plotter = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.conv1 = GCNConv(500, 16)
        self.conv2 = GCNConv(16, 3)

    def forward(self, x):
        edge_index = torch.tensor(self.data['graph']['edge'], dtype=torch.long).t().contiguous()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def train(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        for epoch in range(self.max_epoch):
            optimizer.zero_grad()
            y_pred = self.forward(self.data['graph']['X'])[self.data['train_test_val']['idx_train']]
            train_loss = loss_function(y_pred, self.data['graph']['y'][self.data['train_test_val']['idx_train']])
            train_loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                accuracy_evaluator.data = {
                    'true_y': self.data['graph']['y'][self.data['train_test_val']['idx_train']],
                    'pred_y': y_pred.max(1)[1]
                }
                print(
                    'Epoch:', epoch,
                    'Accuracy:', accuracy_evaluator.evaluate_accuracy(),
                    'Precision', accuracy_evaluator.evaluate_precision(),
                    'Recall', accuracy_evaluator.evaluate_recall(),
                    'F1', accuracy_evaluator.evaluate_f1(),
                    'Loss:', train_loss.item()
                )
                if self.plotter:
                    self.plotter.xs.append(epoch)
                    self.plotter.ys.append(train_loss.item())

    def test(self, X):
        y_pred = self.forward(X)
        return y_pred

    def run(self):
        print('method running...')
        print('--start training...')
        self.train()
        print('--start testing...')
        y_pred = self.test(self.data['graph']['X'])[self.data['train_test_val']['idx_test']]
        return {'pred_y': y_pred.max(1)[1], 'true_y': self.data['graph']['y'][self.data['train_test_val']['idx_test']]}