from abc import ABC, abstractmethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
import numpy as np
from scipy.stats import mode


class DCSTechnique(ABC):
    def __init__(self, n_neighbors=7, algorithm="auto"):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors, algorithm=self.algorithm
        )
        self.current_val_set_X = None
        self.current_val_set_y = None

    def _estimate_competence(self, ensemble, instance):
        pass

    def predict(self, ensemble, instances, real_labels=None):
        competent_members = self._estimate_competence(ensemble, instances)
        predictions_members = np.empty(
            (instances.shape[0], len(ensemble)), dtype=np.int
        )
        for index_clf, clf in enumerate(ensemble):
            predictions_members[:, index_clf] = clf.predict(instances)
        votes = np.zeros(
            (instances.shape[0], np.max(predictions_members) + 1), dtype=np.int
        )
        competent_members[0][0] += 1
        for i in range(predictions_members.shape[0]):
            comp_m = competent_members[i, :]
            np.add.at(votes, (i, predictions_members[i, :]), comp_m)
        return np.argmax(votes, axis=1)

    def fit(self, X, y):
        self.knn = clone(self.knn)
        self.knn.fit(X, y)
        self.current_val_set_X = X
        self.current_val_set_y = y
