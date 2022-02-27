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
    vocab_size = 4727
    embedding_dim = 200
    data = None
    # it defines the max rounds to train the model
    max_epoch = 100
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-4
    sequence_length = 5
    lstm_size = 200
    plotter = None
    word_dict: Dictionary = None

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        self.encoder = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.lstm_size, hidden_size=self.lstm_size, num_layers=1)
        self.fc = nn.Linear(self.lstm_size, self.vocab_size)

    def forward(self, x, prev):
        """Forward propagation"""

        embedded = self.encoder(x)
        out, cur = self.lstm(embedded, prev)
        return self.fc(out), cur

    # backward error propagation will be implemented by pytorch automatically
    # so, we don't need to define the error backpropagation function here

    def train(self, jokes):
        # check for plot setting
        if not self.plotter:
            raise RuntimeWarning("Plotter not defined.")
        if not self.word_dict:
            raise RuntimeWarning("Word Dictionary not defined.")
        
        jokes = list(map(self.word_dict.sentence_to_indexes(200), jokes))
        
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()

        accuracy_evaluator = EvaluateAccuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            cell_state = torch.zeros(1, self.sequence_length, self.lstm_size)
            hidden_state = torch.zeros(1, self.sequence_length, self.lstm_size)
            temp_loss = None
            for batch, X in enumerate(jokes):
                optimizer.zero_grad()
                batch_X = []
                batch_y = []
                for i in range(0, len(X) // self.sequence_length - 1):
                    start = i * self.sequence_length
                    batch_X.append(X[start:start+self.sequence_length])
                    batch_y.append(X[start+self.sequence_length:start+self.sequence_length+self.sequence_length])
                X_train = torch.tensor(np.array(batch_X)).to(device)
                y_pred, (hidden_state, cell_state) = self.forward(X_train, (hidden_state, cell_state))
                y_true = torch.LongTensor(np.array(batch_y)).to(device)

                train_loss = loss_function(y_pred.transpose(1, 2), y_true)
                hidden_state = hidden_state.detach()
                cell_state = cell_state.detach()

                train_loss.backward()
                optimizer.step()
                print('epoch', epoch, 'batch', batch, 'loss', train_loss.item())

            print('–----------------–-----------------------------------')

            # add to graph if avaliable
        if self.plotter:
            self.plotter.xs.append(epoch)
            self.plotter.ys.append(train_loss.item())
    def predict(self, text):
        words = text.split(' ')
        cell_state = torch.zeros(1, 3, self.lstm_size)
        hidden_state = torch.zeros(1, 3, self.lstm_size)
        count = 0
        next_word = None
        while next_word != '.' and count < 100:
            x = torch.tensor([[self.word_dict.dictionary[w] for w in words[count:]]])
            y_pred, (hidden_state, cell_state) = self.forward(x, (hidden_state, cell_state))
            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            next_word = self.word_dict.inverse_dictionary[word_index]
            words.append(next_word)
            count += 1
        return words

    def run(self):
        print('method running...')
        print('--start training...')

        self.train(self.data)
        text = None
        while text != 'stop':
            text = input('enter three words separated by space')
            print(self.predict(text))

