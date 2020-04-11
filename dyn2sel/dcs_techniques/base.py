from abc import ABC, abstractmethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone


class DCSTechnique(ABC):
    def __init__(self, n_neighbors=7, algorithm="auto"):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, algorithm=self.algorithm)
        self.current_val_set_X = None
        self.current_val_set_y = None

    @abstractmethod
    def estimate_competence(self, ensemble, instance):
        pass

    def fit(self, X, y):
        self.knn = clone(self.knn)
        self.knn.fit(X, y)
        self.current_val_set_X = X
        self.current_val_set_y = y
