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
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from sklearn.utils import shuffle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MethodRNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-4
    # no larger than 100!!!!!!!!!!!!!!!!
    plotter = None

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription,
                vocab_size, embedding_dim, hidden_dim, n_layers,
                bidirec, dropout, pad_idx, unk_idx):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # dimension, number of embedding
        self.encoder = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirec,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.activ = nn.Softmax(dim=1)
        
        self.encoder.weight.data[pad_idx] = torch.zeros(embedding_dim)
        self.encoder.weight.data[unk_idx] = torch.zeros(embedding_dim)

    def forward(self, x, lengths):
        """Forward propagation"""
        
        embedded = self.encoder(x)
        embedded = self.dropout(embedded)
        
        packed_embedded = pack_padded_sequence(embedded, lengths)
        packed_out, (hidden, _) = self.lstm(packed_embedded)

        out, out_lengths = pad_packed_sequence(packed_out)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        
        y_pred = self.activ(self.fc(hidden))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so, we don't need to define the error backpropagation function here

    def train(self, batches):
        
        # check for plot setting
        if not self.plotter:
            raise RuntimeWarning("Plotter not defined.")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # loss_function = nn.CrossEntropyLoss().to(device)
        loss_function = nn.BCEWithLogitsLoss().to(device)
        accuracy_evaluator = EvaluateAccuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            loss_acc = np.zeros(5)
            for batch in batches:
                optimizer.zero_grad()
                X_train, train_len = batch.text
                y_true = batch.label
                
                y_pred = self.forward(X_train, train_len.cpu()).squeeze(1)
                # calculate the training loss
                train_loss = loss_function(y_pred, y_true)

                train_loss.backward()
                optimizer.step()
                
                accuracy_evaluator.data = {
                    'true_y': y_true.cpu().detach().numpy(),
                    'pred_y': y_pred.cpu().detach().numpy()
                }
                loss_acc += np.array([
                    train_loss.item(),
                    accuracy_evaluator.evaluate_accuracy(),
                    accuracy_evaluator.evaluate_precision(),
                    accuracy_evaluator.evaluate_recall(),
                    accuracy_evaluator.evaluate_f1()
                ])

            loss_acc = loss_acc / len(batch) 
            print('------------------------------------------------------------')
            print('Epoch:', epoch)
            print('evaluating accuracy performance...')
            print('Accuracy:', loss_acc[1])
            print('evaluating precision performance...')
            print('Precision', loss_acc[2])
            print('evaluating recall performance...')
            print('Recall', loss_acc[3])
            print('evaluating f1 performance...')
            print('F1', loss_acc[4])
            print('Loss:', loss_acc[0])
            print('------------------------------------------------------------')
                
            # add to graph if avaliable
            if self.plotter:
                self.plotter.xs.append(epoch)
                self.plotter.ys.append(loss)

    def test(self, batches):
        y_pred = None
        y_true = None
        for batch in batches:
            X_test, test_len = batch.text
            if i == 0:
                y_pred = self.forward(X_test, test_len)
                y_true = torch.tensor(batch.label)
            else:
                y_pred = torch.cat((y_pred, self.forward(X_test, test_len.cpu())), 0)
                y_true = torch.cat((y_true, batch.label), 0)
        return y_true.cpu(), y_pred.cpu()

    def run(self):
        print('method running...')
        print('--start training...')
        
        self.train(self.data['train'])
        print('--start testing...')
        ture_y, pred_y = self.test(self.data['test'])
        return {'pred_y': pred_y, 'true_y': true_y}
