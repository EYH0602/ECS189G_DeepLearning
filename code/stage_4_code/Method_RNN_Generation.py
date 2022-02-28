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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MethodRNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-4
    plotter = None
    
    lstm_size = 128
    embedding_dim = 128
    num_layers = 3

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, vocab_size, sequence_length):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        self.sequence_length = sequence_length
        
        # dimension, number of embedding
        self.encoder = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2
        )
        self.fc = nn.Linear(self.lstm_size, vocab_size)
        # self.dropout = nn.Dropout(0.2)
        # self.activ = nn.LogSigmoid()
        
    def forward(self, x, lengths):
        """Forward propagation"""
        
        embedded = self.encoder(x)
        # embedded = self.dropout(embedded)
        
        output, state = self.lstm(embedded, lengths)

        # out, out_lengths = pad_packed_sequence(packed_out)
        # hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        
        logits = self.fc(output)
        return logits, state
    
    def init_state(self, sequence_length):
        return (
            torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(device),
            torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(device)
        )

    # backward error propagation will be implemented by pytorch automatically
    # so, we don't need to define the error backpropagation function here

    def train(self, data):
        
        # check for plot setting
        if not self.plotter:
            raise RuntimeWarning("Plotter not defined.")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss().to(device)
        # loss_function = nn.BCEWithLogitsLoss().to(device)

        for epoch in range(self.max_epoch):
            loss = 0
            state_h, state_c = self.init_state(self.sequence_length)
            for batch, (X_train, y_true) in enumerate(data):
                optimizer.zero_grad()
                
                X_train = X_train.to(device)
                y_true = y_true.to(device)
                
                y_pred, (state_h, state_c) = self.forward(X_train, (state_h, state_c))
                
                # calculate the training loss
                train_loss = loss_function(y_pred.transpose(1, 2), y_true)
                
                state_h = state_h.detach()
                state_c = state_c.detach()
                
                train_loss.backward()
                optimizer.step()
                
                loss += train_loss.item()

            loss = loss / len(data) 
            if epoch % 50 == 0:
                print('------------------------------------------------------------')
                print('Epoch:', epoch)
                print('Loss:', loss)
                print('------------------------------------------------------------')
                
            # add to graph if avaliable
            if self.plotter:
                self.plotter.xs.append(epoch)
                self.plotter.ys.append(loss)

    def test(self, batches):
        accuracy_evaluator = EvaluateAccuracy('training evaluator', '')
        acc_scores = []

        for batch in batches:
            X_test, test_len = batch.text
            y_pred = torch.round(self.activ(self.forward(X_test, test_len.cpu())))
            y_true = batch.label
            accuracy_evaluator.data = {
                'true_y': y_true.cpu(),
                'pred_y': y_pred.cpu().detach().numpy()
            }
            acc_scores.append(accuracy_evaluator.evaluate_accuracy())
        return np.mean(acc_scores), np.std(acc_scores)
    
    def predict(self, text, next_words=100):
        words = text.split(' ')
        state_h, state_c = self.init_state(len(words))
        
        for i in range(next_words):
            X = torch.tensor([[self.dataset.word_to_index[w] for w in words[i:]]]).to(device)
            y_pred, (state_h, state_c) = self.forward(X, (state_h, state_c))
            
            last_word_logits = y_pred[0][-1]
            p = F.softmax(last_word_logits, dim=0).cpu().detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            words.append(self.dataset.index_to_word[word_index])
        
        return words

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data)
        torch.save(self.state_dict(), 'stage_4_generation.pt')

        # print('--start testing...')
        # test_result = self.test(self.data['test'])
        # return test_result
