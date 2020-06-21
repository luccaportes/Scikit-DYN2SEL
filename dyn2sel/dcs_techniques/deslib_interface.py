from dyn2sel.dcs_techniques import DCSTechnique
from abc import abstractmethod


class DESLIBInterface(DCSTechnique):
    def __init__(self, n_neighbors=7, algorithm="auto"):
        super().__init__(n_neighbors=n_neighbors, algorithm=algorithm)
        self.deslib_stencil = self.set_stencil()
        self.deslib_alg = None
        self.needs_fitting = True

    @abstractmethod
    def set_stencil(self):
        pass

    def fit(self, X, y):
        self.current_val_set_X = X
        self.current_val_set_y = y
        self.needs_fitting = True

    def predict(self, ensemble, instances):
        if self.needs_fitting:
            self.deslib_alg = self.deslib_stencil(ensemble, k=self.n_neighbors)
            self.deslib_alg.fit(self.current_val_set_X, self.current_val_set_y)
            self.needs_fitting = False
        return self.deslib_alg.predict(instances)

    def estimate_competence(self, ensemble, instance):
        pass
