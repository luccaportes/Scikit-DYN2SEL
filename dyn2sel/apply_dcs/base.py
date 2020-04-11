from abc import abstractmethod

from skmultiflow.core import ClassifierMixin


class DCSApplier(ClassifierMixin):

    @abstractmethod
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass
