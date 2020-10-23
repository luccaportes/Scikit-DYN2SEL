from skmultiflow.core import BaseSKMObject, ClassifierMixin


class skmultiflow_encapsulator(BaseSKMObject, ClassifierMixin):
    def __init__(self, clf):
        self.clf = clf

    @property
    def classes_(self):
        try:
            return self.clf.classes
        except AttributeError:
            try:
                return self.clf.classes_
            except AttributeError:
                return self.clf._classes

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        self.clf.partial_fit(X, y, classes=classes, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
