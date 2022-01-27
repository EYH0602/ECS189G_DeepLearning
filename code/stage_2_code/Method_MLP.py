"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import EvaluateAccuracy
import torch
from torch import nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("training on: " + device.type)


class MethodMLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 1000
    except_accuracy = 0.995
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-4
    plotter = None

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 512)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()

        self.fc_layer_2 = nn.Linear(512, 512)
        self.activation_func_2 = nn.ReLU()

        self.fc_layer_3 = nn.Linear(512, 10)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_3 = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        """Forward propagation"""
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))
        h = self.activation_func_2(self.fc_layer_2(h))
        # output layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_3(self.fc_layer_3(h))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so, we don't need to define the error backpropagation function here

    def train(self, X, y):
        
        # check for plot setting
        if not self.plotter:
            raise RuntimeWarning("Plotter not defined.")
        
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = EvaluateAccuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        # you can do an early stop if self.max_epoch is too much...
        for epoch in range(self.max_epoch):

            # gradient optimizer need to be clear every epoch
            optimizer.zero_grad()

            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it

            X_train = torch.FloatTensor(np.array(X))
            X_train.to(device)  # use cuda if available
            y_pred = self.forward(X_train)
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch % 100 == 0:
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
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
