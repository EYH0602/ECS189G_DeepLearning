"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import EvaluateAccuracy
from code.stage_4_code.Dictionary import Dictionary
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MethodRNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 1000
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-4
    plotter = None
    word_dict: Dictionary = None

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, embedding_dim):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.embedding_dim = embedding_dim
        
        self.encoder = nn.Embedding(2000, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, 50)
        self.fc = nn.Linear(50, 2)
        self.activation = nn.Sigmoid()

    def forward(self, x, lengths):
        """Forward propagation"""
        
        embedded = self.encoder(x)
        packed_embedded = pack_padded_sequence(embedded, lengths)
        packed_out, _ = self.lstm(packed_embedded)
        out, _ = pad_packed_sequence(packed_out)
        
        row_indices = torch.arange(0, x.size(0)).long()
        col_indices = lengths - 1
        
        out_tensor = out[row_indices, col_indices, :]
        
        y_pred = self.activation(self.fc(out_tensor))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so, we don't need to define the error backpropagation function here

    def train(self, X, y):
        
        # check for plot setting
        if not self.plotter:
            raise RuntimeWarning("Plotter not defined.")
        if not self.word_dict:
            raise RuntimeWarning("Word Dictionary not defined.")
        
        X = list(map(self.word_dict.sentence_to_indexes(self.embedding_dim), X))
        train_len = torch.tensor(list(map(len, X)))
        
        X_train = torch.tensor(np.array(X)).to(device)
        y_true = torch.tensor(np.array(y)).to(device)
        
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

            y_pred = self.forward(X_train, train_len)

            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            train_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                accuracy_evaluator.data = {
                    'true_y': y_true.cpu(),
                    'pred_y': y_pred.cpu().max(1)[1]
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
        X = list(map(self.word_dict.sentence_to_indexes(self.embedding_dim), X))
        test_len = torch.tensor(list(map(len, X)))
        X = torch.tensor(np.array(X)).to(device)
        y_pred = self.forward(X, test_len)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.cpu().max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
