'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from src.base_class.method import method
from sklearn import svm


class Method_SVM(method):
    c = None
    data = None
    
    def train(self, X, y):
        # check here for the svm.SVC doc: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        model = svm.SVC(C = self.c)
        # model variable learning has been built-in by the fit() function
        # check here for the svm.fit doc: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.fit
        model.fit(X, y)
        return model
    
    def test(self, model, X):
        # check here for the svm.predict doc: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.predict
        return model.predict(X)
    
    def run(self):
        print('method running...')
        print('--start training...')
        model = self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(model, self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            