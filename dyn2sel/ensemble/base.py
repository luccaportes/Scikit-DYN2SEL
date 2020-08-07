from abc import ABC, abstractmethod


class Ensemble(ABC):
    def __init__(self):
        self.classes_ = None
        self.ensemble = []

    def __iter__(self):
        for i in self.ensemble:
            yield i

    def __len__(self):
        return len(self.ensemble)

    def __getitem__(self, key):
        for i in range(len(self.ensemble)):
            if i == key:
                return self.ensemble[i]
        raise IndexError

    @abstractmethod
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def add_member(self, clf):
        pass

    @abstractmethod
    def del_member(self, index=-1):
        pass
