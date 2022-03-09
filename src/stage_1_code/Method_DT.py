'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from src.base_class.method import method
from sklearn import tree


class Method_DT(method):
    c = None
    data = None
    
    def train(self, X, y):
        # check here for the decision tree classifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
        model = tree.DecisionTreeClassifier()
        # check here for decision tree fit doc: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.fit
        model.fit(X, y)
        return model
    
    def test(self, model, X):
        # check here for decision tree predict doc: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.predict
        return model.predict(X)
    
    def run(self):
        print('method running...')
        print('--start training...')
        model = self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(model, self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            