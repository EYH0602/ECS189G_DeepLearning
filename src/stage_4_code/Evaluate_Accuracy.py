"""
Concrete Evaluate class for a specific evaluation metrics
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from src.base_class.evaluate import evaluate
from sklearn.metrics import \
    accuracy_score, precision_score, recall_score, f1_score


class EvaluateAccuracy(evaluate):
    data = None

    def evaluate_accuracy(self):
        #print('evaluating accuracy performance...')
        return accuracy_score(self.data['true_y'], self.data['pred_y'])

    def evaluate_precision(self):
        #print('evaluating precision performance...')
        return precision_score(self.data['true_y'], self.data['pred_y'], average="micro")

    def evaluate_recall(self):
        #print('evaluating recall performance...')
        return recall_score(self.data['true_y'], self.data['pred_y'], average="micro")
    
    def evaluate_f1(self):
        #print('evaluating f1 performance...')
        return f1_score(self.data['true_y'], self.data['pred_y'], average="micro")
