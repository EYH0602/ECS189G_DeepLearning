"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import EvaluateAccuracy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("training on: " + device.type)


class MethodCNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 300
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    plotter = None

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture
    # should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc_layer_1 = nn.Linear(16 * 4 * 4, 120)
        self.activation_func_1 = nn.ReLU()

        self.fc_layer_2 = nn.Linear(120, 10)
        self.activation_func_2 = nn.Softmax(dim=1)
        
        if torch.cuda.is_available():
            self.conv1 = self.conv1.cuda()
            self.conv2 = self.conv2.cuda()
            self.fc_layer_1 = self.fc_layer_1.cuda()
            self.fc_layer_2 = self.fc_layer_2.cuda()

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        """Forward propagation"""
        # CNN  layers
        if torch.cuda.is_available():
            h = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)).cuda()
            h = F.max_pool2d(F.relu(self.conv2(h)), 2).cuda()
        else:
            h = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            h = F.max_pool2d(F.relu(self.conv2(h)), 2)

        # FC layers
        h = torch.flatten(h)
        h = self.activation_func_1(self.fc_layer_1(h))
        y_pred = self.fc_layer_2(h)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so, we don't need to define the error backpropagation function here

    def train(self):

        # check for plot setting
        if not self.plotter:
            raise RuntimeWarning("Plotter not defined.")

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = EvaluateAccuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        for epoch in range(self.max_epoch):

            y_true = []
            y_pred = []
            print(epoch)

            # gradient optimizer need to be clear every epoch
            optimizer.zero_grad()

            for data in self.data['train']:
                X = data['image']
                y_true.append(data['label'])

                # add dimension to input
                # 28 x 28 -> 1 x 1 x 28 x 28
                X_train = torch.FloatTensor(
                    np.array(X)).unsqueeze(0).unsqueeze(0)

                X_train = X_train.to(device)  # use cuda if available
                y_pred.append(self.forward(X_train))

            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y_true)).to(device)
            y_pred = self.activation_func_2(torch.stack(y_pred).to(device))

            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            train_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                accuracy_evaluator.data = {
                    'true_y': y_true.cpu(),
                    'pred_y': y_pred.max(1)[1].cpu()
                }
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
                    self.plotter.xs.append(epoch)
                    self.plotter.ys.append(train_loss.item())

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(X)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred

    def run(self):
        print('method running...')
        print('--start training...')
        self.train()
        print('--start testing...')

        y_pred = []
        y_true = []

        for test_data in self.data['test']:
            X = torch.FloatTensor(np.array(test_data['image']))
            X = X.unsqueeze(0).unsqueeze(0)
            y_true.append(test_data['label'])
            y_pred.append(self.test(X))

        return {
            'pred_y': self.activation_func_2(torch.stack(y_pred)).max(1)[1],
            'true_y': y_true
        }
