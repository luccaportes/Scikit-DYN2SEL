from abc import abstractmethod

from skmultiflow.core import ClassifierMixin


class DCSApplier(ClassifierMixin):

    @abstractmethod
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        pass

    @abstractmethod
    def predict(self, X, y=None):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X, y), sample_weight=sample_weight)

    def is_oracle(self):
        return self.dcs_method.is_oracle()
