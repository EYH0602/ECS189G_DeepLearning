"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from random import shuffle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("training on: " + device.type)


class Method_CNN(method, nn.Module):

    data = None
    # it defines the max rounds to train the model
    max_epoch = 1
    except_accuracy = 0.995
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-4
    plotter = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=1, padding=0)

        self.fc_layer_1 = nn.Linear(5 * 5 * 10, 125)
        self.activation_func_1 = nn.ReLU()

        self.fc_layer_2 = nn.Linear(125, 10)
        self.activation_func_2 = nn.Softmax(dim=1)
    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        """Forward propagation"""
        # Covolutional layers
        h = F.max_pool2d(F.relu(self.conv1(x)), 2, stride=2)
        h = F.max_pool2d(F.relu(self.conv2(h)), 2, stride=2)

        # FC layers
        h = torch.flatten(h)
        h = self.activation_func_1(self.fc_layer_1(h))
        y_pred = self.fc_layer_2(h)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so, we don't need to define the error backpropagation function here

    def train(self, train_data):
        # check for plot setting
        if not self.plotter:
            raise RuntimeWarning("Plotter not defined.")

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        # you can do an early stop if self.max_epoch is too much...
        batch_size = 256
        for epoch in range(self.max_epoch):
            print(epoch)
            shuffle(train_data)
            mini_batches = []

            for i in range(0, len(train_data) // batch_size - 1):
                start_index = i * batch_size
                mini_batch = train_data[start_index:start_index + batch_size - 1]
                if len(mini_batch) == 0:
                    break
                mini_batches.append({'X': [pair['image'] for pair in mini_batch], 'y': [pair['label'] for pair in mini_batch]})
            for mini_batch in mini_batches:
                optimizer.zero_grad()
                y_pred = []
                X = mini_batch['X']
                y = mini_batch['y']
                # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
                for x in X:
                    X_train = torch.FloatTensor(np.array(x)).permute(2, 0, 1).unsqueeze(0)
                    y_pred.append(self.forward(X_train))
                X_train.to(device)  # use cuda if available

                # convert y to torch.tensor as well
                y_true = torch.LongTensor(np.array(y))
                y_pred = self.activation_func_2(torch.stack(y_pred))
                # calculate the training loss
                train_loss = loss_function(y_pred, y_true)

                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()

            if epoch % 1 == 0:
                accuracy_evaluator.data = {
                    'true_y': y_true,
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
        self.train(self.data['train'])
        print('--start testing...')
        y_pred = []
        for test_data in self.data['test']['X']:
            X = torch.FloatTensor(np.array(test_data)).permute(2, 0, 1).unsqueeze(0)
            y_pred.append(self.test(X))
        return {'pred_y': self.activation_func_2(torch.stack(y_pred)).max(1)[1], 'true_y': self.data['test']['y']}